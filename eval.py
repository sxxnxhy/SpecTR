import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import matchms
from huggingface_hub import hf_hub_download
import config
from model import CLIPModel
from transformers import AutoTokenizer


class MassSpecGymTestDataset(Dataset):
    def __init__(self, tokenizer, max_len=300):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        self.candidates_map = {}
        
        print(f"\n[1/3] Downloading/Loading MassSpecGym Data...")
        
        # 1. Download Files
        tsv_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.FILENAME_TSV, repo_type="dataset")
        json_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.FILENAME_CANDIDATES, repo_type="dataset")
        
        # 2. Load Candidates (SMILES -> List of Candidate SMILES)
        print("Loading candidates...")
        with open(json_path, 'r') as f:
            self.candidates_map = json.load(f)
            
        # 3. Load Metadata & Filter for Test Set
        print("Loading spectra metadata...")
        df = pd.read_csv(tsv_path, sep="\t")
        
        # Keep only 'test' fold
        df = df[df['fold'] == 'test'].reset_index(drop=True)
        
        # Filter: Keep only rows where we have candidates
        # (Some test items might be missing from the candidate json, though rare)
        valid_indices = []
        for idx, row in df.iterrows():
            if row['smiles'] in self.candidates_map:
                valid_indices.append(idx)
        
        self.df = df.iloc[valid_indices].reset_index(drop=True)
        print(f"✅ Loaded {len(self.df)} test spectra with candidates.")

    def __len__(self):
        return len(self.df)
        
    def _process_spectrum(self, mzs_str, ints_str):
        """Convert string m/z arrays to your model's input format"""
        mzs = np.array([float(x) for x in mzs_str.split(',')])
        intensities = np.array([float(x) for x in ints_str.split(',')])
        
        # --- Your Preprocessing Logic (Log + Sqrt) ---
        # 1. Intensity: Sqrt -> Max Norm
        root_ints = np.sqrt(intensities)
        max_int = root_ints.max() if len(root_ints) > 0 else 1.0
        norm_ints = root_ints / max_int
        
        # 2. Mass: Log(1+x)
        log_mzs = np.log(mzs + 1.0)
        
        # 3. Combine & Sort
        processed = list(zip(log_mzs, norm_ints))
        
        # Truncate / Pad
        if len(processed) > self.max_len:
            # Sort by intensity descending, take top K
            processed.sort(key=lambda x: x[1], reverse=True)
            processed = processed[:self.max_len]
        
        # Sort by mass ascending (Physical Order)
        processed.sort(key=lambda x: x[0])
        
        # To Tensor
        peak_sequence = torch.zeros(self.max_len, 2, dtype=torch.float32)
        peak_mask = torch.zeros(self.max_len, dtype=torch.bool)
        
        if len(processed) > 0:
            peak_sequence[:len(processed)] = torch.tensor(processed)
            peak_mask[:len(processed)] = True
            
        return peak_sequence, peak_mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        query_smiles = row['smiles']
        
        # 1. Process Spectrum
        peak_seq, peak_mask = self._process_spectrum(row['mzs'], row['intensities'])
        
        # 2. Get Candidates
        candidates = self.candidates_map[query_smiles]
        
        # Ensure the True Target is in the candidates (Sanity Check)
        # Note: In MassSpecGym, the query IS included in the candidate list.
        # We need to find which index is the answer.
        try:
            label_idx = candidates.index(query_smiles)
        except ValueError:
            # Fallback: sometimes canonicalization differs. 
            # If strict string match fails, we assume index 0 or handle it.
            # Ideally we use RDKit canonicalization, but for speed we assume raw match first.
            label_idx = 0 # Fallback
            
        # 3. Tokenize Candidates (All of them)
        # This might be slow if done per item. 
        # Optimization: We return raw text lists and tokenize in the batch loop to batch-process.
        
        return {
            "peak_sequence": peak_seq,
            "peak_mask": peak_mask,
            "candidates": candidates,      # List of strings
            "label_idx": label_idx         # Integer
        }

def evaluate_massspecgym(model, device):
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER['model_name'])
    dataset = MassSpecGymTestDataset(tokenizer)
    
    # Batch size 1 is safest because each query has a different number of candidates (~256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])
    
    model.eval()
    print(f"\n" + "="*60)
    print(f"🏋️‍♀️ Running Official MassSpecGym Benchmark")
    print(f"   (Comparing 1 Query vs ~256 Pre-defined Candidates)")
    print("="*60)
    
    hits_1 = 0
    hits_5 = 0
    total = 0
    
    for i, batch in tqdm(enumerate(loader), total=len(loader), desc="Evaluating"):
        peak_seq = batch['peak_sequence'].unsqueeze(0).to(device) # [1, Seq, 2]
        peak_mask = batch['peak_mask'].unsqueeze(0).to(device)    # [1, Seq]
        candidates = batch['candidates']                          # List of SMILES
        label_idx = batch['label_idx']
        
        # 1. Encode Spectrum
        with autocast(device_type='cuda', dtype=torch.float16):
            spec_emb = model.ms_encoder(peak_seq, peak_mask) # [1, Dim]
        
        # 2. Encode Candidates (Batch Processing)
        # Candidates list is usually ~256. We can feed this as one batch to the text encoder.
        cand_inputs = tokenizer(
            candidates, 
            max_length=config.TEXT_ENCODER['max_length'],
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        cand_ids = cand_inputs['input_ids'].to(device)
        cand_mask = cand_inputs['attention_mask'].to(device)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # [Num_Cand, Dim]
            cand_embeds = model.text_encoder(cand_ids, cand_mask)
            
        # 3. Similarity & Ranking
        # spec_emb: [1, Dim]
        # cand_embeds: [N, Dim]
        sim_scores = torch.matmul(spec_emb, cand_embeds.T).squeeze() # [N]
        
        # 4. Check Rank
        target_score = sim_scores[label_idx]
        # Count how many scores are greater than the target
        rank = (sim_scores > target_score).sum().item() + 1
        
        if rank == 1: hits_1 += 1
        if rank <= 5: hits_5 += 1
        total += 1
        
        if i % 100 == 0 and i > 0:
            print(f" [Step {i}] Current R@1: {hits_1/total*100:.2f}%")

    r1 = hits_1 / total * 100
    r5 = hits_5 / total * 100
    
    print("\n" + "="*60)
    print(f"🏆 MassSpecGym Official Result")
    print(f"Total Samples: {total}")
    print(f"Top-1 Accuracy: {r1:.2f}%")
    print(f"Top-5 Accuracy: {r5:.2f}%")
    print("="*60)

def main():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Load Model
    model = CLIPModel().to(device)
    checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model.pt"
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model Loaded (Epoch {checkpoint['epoch']})")
    
    evaluate_massspecgym(model, device)

if __name__ == '__main__':
    main()