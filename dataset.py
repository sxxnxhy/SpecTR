import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
import config
import json
import pickle
# [REMOVED] RDKit imports are gone
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- HELPER FUNCTION (Physics Only - No Chemistry/SMILES logic) ---

def process_spectrum_data(args):
    """
    Worker function to process raw strings into Tensors ONCE.
    Pure physics (Math), no RDKit.
    """
    mzs_str, ints_str, max_len = args
    
    try:
        mzs = np.array([float(x) for x in mzs_str.split(',')])
        intensities = np.array([float(x) for x in ints_str.split(',')])
    except:
        return torch.zeros(max_len, 2), torch.zeros(max_len, dtype=torch.bool)

    # Square root -> Max Norm -> Log Mass
    root_ints = np.sqrt(intensities)
    max_int = root_ints.max() if len(root_ints) > 0 else 1.0
    if max_int < 1e-9: max_int = 1.0
    norm_ints = root_ints / max_int
    log_mzs = np.log(mzs + 1.0)
    
    processed = list(zip(log_mzs, norm_ints))
    
    # Sort by Intensity (Keep top K)
    if len(processed) > max_len:
        processed.sort(key=lambda x: x[1], reverse=True)
        processed = processed[:max_len]
    
    # Sort by Mass (Positional)
    processed.sort(key=lambda x: x[0])
    
    peak_sequence = torch.zeros(max_len, 2, dtype=torch.float32)
    peak_mask = torch.zeros(max_len, dtype=torch.bool)
    
    if len(processed) > 0:
        peak_sequence[:len(processed)] = torch.tensor(processed)
        peak_mask[:len(processed)] = True
        
    return peak_sequence, peak_mask

# -----------------------------------------------------------------------

class MassSpecGymTrainDataset(Dataset):
    def __init__(self, tokenizer, split, max_len=300, retrieval_mode=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.retrieval_mode = retrieval_mode
        
        if isinstance(split, str): self.split_names = [split]
        else: self.split_names = split
            
        split_sig = "_".join(self.split_names)
        print(f"\n[Data] Initializing MassSpecGym for {self.split_names} (Retrieval: {retrieval_mode})")

        # --- 1. CACHE CHECK (RAW Version) ---
        cache_filename = f"cached_raw_no_rdkit_{split_sig}_{retrieval_mode}.pkl"
        
        if os.path.exists(cache_filename):
            print(f"   -> 🚀 Fast-loading TENSORS from cache: {cache_filename}")
            with open(cache_filename, 'rb') as f:
                data = pickle.load(f)
                self.df = data['df']
                self.candidates_map = data.get('candidates_map', {})
                self.precomputed_spectra = data['spectra'] 
            print(f"   -> ✅ Loaded {len(self.df)} samples.")
            return

        # --- 2. LOAD & PROCESS FROM SCRATCH ---
        print("   -> ⏳ Processing data from scratch (Raw Mode - No RDKit)...")
        
        # Load TSV
        try:
            tsv_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.FILENAME_TSV, repo_type="dataset")
        except:
            tsv_path = config.FILENAME_TSV
            
        df = pd.read_csv(tsv_path, sep="\t")
        self.df = df[df['fold'].isin(self.split_names)].reset_index(drop=True)
        self.df = self.df.dropna(subset=['smiles'])
        
        # [REMOVED] No SMILES processing. We trust the raw string.
        
        # --- PRE-COMPUTE TENSORS (Physics) ---
        print(f"   -> ⚡ Pre-computing Spectra Tensors (CPU Optimized)...")
        tasks = [
            (row['mzs'], row['intensities'], self.max_len) 
            for _, row in self.df.iterrows()
        ]
        
        with ProcessPoolExecutor() as executor:
            self.precomputed_spectra = list(tqdm(
                executor.map(process_spectrum_data, tasks), 
                total=len(tasks),
                desc="Processing Spectra"
            ))

        # 3. Load Candidates (Validation Only)
        self.candidates_map = {}
        if self.retrieval_mode:
            print(f"   -> Loading Candidates JSON...")
            try:
                json_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.FILENAME_CANDIDATES, repo_type="dataset")
            except:
                json_path = config.FILENAME_CANDIDATES
            
            # [RAW MODE] Just load the JSON directly. No processing.
            with open(json_path, 'r') as f:
                self.candidates_map = json.load(f)
            
            # Filter DF: We check if the raw SMILES string exists as a key in the JSON
            # WARNING: If there is a "typo" difference (e.g. [H]), this will filter out the data.
            valid_indices = [i for i, row in self.df.iterrows() if row['smiles'] in self.candidates_map]
            
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
            self.precomputed_spectra = [self.precomputed_spectra[i] for i in valid_indices]
            
            print(f"   -> Filtered to {len(self.df)} queries.")

        # --- 4. SAVE CACHE ---
        print(f"   -> 💾 Saving TENSOR cache to {cache_filename}...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({
                'df': self.df, 
                'candidates_map': self.candidates_map,
                'spectra': self.precomputed_spectra
            }, f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Array lookup (Instant)
        peak_seq, peak_mask = self.precomputed_spectra[idx]
        row = self.df.iloc[idx]
        smiles = row['smiles']
        
        item = {
            'peak_sequence': peak_seq,
            'peak_mask': peak_mask,
        }

        if self.retrieval_mode:
            item['smiles'] = smiles
            # Direct lookup from raw map
            item['candidates_smiles'] = self.candidates_map[smiles] 
        else:
            tokenized_text = self.tokenizer(
                str(smiles),
                max_length=config.TEXT_ENCODER['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item['input_ids'] = tokenized_text['input_ids'].squeeze(0)
            item['attention_mask'] = tokenized_text['attention_mask'].squeeze(0)
            
        return item

if __name__ == "__main__":
    from transformers import AutoTokenizer
    import random
    
    print("--- RUNNING RAW DATASET CHECK (NO RDKIT) ---")
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER['model_name'])
    
    # Init in Val mode
    ds = MassSpecGymTrainDataset(tokenizer, split='train', retrieval_mode=False)
    
    if len(ds) == 0:
        print("\n❌ CRITICAL ERROR: Dataset is empty after filtering!")
        print("   Reason: The SMILES strings in the TSV file do NOT match the Keys in the JSON file.")
        print("   Solution: You MUST put RDKit back, or the files are incompatible as raw text.")
    else:
        print(f"\n✅ Success: {len(ds)} samples matched between TSV and JSON.")
        
        # Check one sample
        idx = random.randint(0, len(ds)-1)
        item = ds[idx]
        gt = item['smiles']
        cands = item['candidates_smiles']
        
        print(f"Sample [{idx}]:")
        print(f" - GT: {gt}")
        print(f" - Candidates: {len(cands)}")
        
        if gt in cands:
             print(" - ✅ Ground Truth found in candidates (String Exact Match).")
        else:
             print(" - ❌ WARNING: Ground Truth NOT found in candidates!")
             print("   Reason: The GT string format differs from the Candidate list string format.")