import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
import config


class MassSpecGymTrainDataset(Dataset):
    """
    Parses MassSpecGym.tsv for Training/Validation.
    """
    def __init__(self, tokenizer, split, max_len=300):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        print(f"\n[Data] Loading MassSpecGym ({split} set)...")
        
        # Download/Load
        try:
            tsv_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.FILENAME_TSV, repo_type="dataset")
        except:
            # Fallback if local file exists
            tsv_path = config.FILENAME_TSV
            
        df = pd.read_csv(tsv_path, sep="\t")
        
        # Filter by Split (fold column)
        self.df = df[df['fold'] == split].reset_index(drop=True)
        
        # Remove rows with empty smiles
        self.df = self.df.dropna(subset=['smiles'])
        print(f" Loaded {len(self.df)} spectra for {split}.")

    def __len__(self):
        return len(self.df)
    
    def _process_spectrum(self, mzs_str, ints_str):
        # Parse comma-separated strings
        try:
            mzs = np.array([float(x) for x in mzs_str.split(',')])
            intensities = np.array([float(x) for x in ints_str.split(',')])
        except:
            return torch.zeros(self.max_len, 2), torch.zeros(self.max_len, dtype=torch.bool)

        # --- YOUR PHYSICS PREPROCESSING ---
        # 1. Intensity: Sqrt -> Max Norm
        root_ints = np.sqrt(intensities)
        max_int = root_ints.max() if len(root_ints) > 0 else 1.0
        if max_int < 1e-9: max_int = 1.0
        norm_ints = root_ints / max_int
        
        # 2. Mass: Log(1+x)
        log_mzs = np.log(mzs + 1.0)
        
        # 3. Combine
        processed = list(zip(log_mzs, norm_ints))
        
        # 4. Truncate (Top-K Intensity) & Sort (Mass)
        if len(processed) > self.max_len:
            processed.sort(key=lambda x: x[1], reverse=True)
            processed = processed[:self.max_len]
        
        # Sort by mass for Transformer positional consistency
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
        smiles = row['smiles']
        
        # 1. Tokenize Text
        tokenized_text = self.tokenizer(
            str(smiles),
            max_length=config.TEXT_ENCODER['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. Process Spectrum
        peak_seq, peak_mask = self._process_spectrum(row['mzs'], row['intensities'])
        
        return {
            'peak_sequence': peak_seq,
            'peak_mask': peak_mask,
            'input_ids': tokenized_text['input_ids'].squeeze(0),
            'attention_mask': tokenized_text['attention_mask'].squeeze(0)
        }
        
        
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from tqdm import tqdm
    import config

    print("--- RUNNING DATASET SAFETY CHECK ---")
    
    # 1. Initialize Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER['model_name'])
    
    # 2. Load Dataset
    ds = MassSpecGymTrainDataset(tokenizer, split='train')
    
    # 3. Check Variables
    smiles_limit = config.TEXT_ENCODER['max_length']
    peaks_limit = config.MAX_PEAK_SEQ_LEN
    
    smiles_violation_count = 0
    max_smiles_len = 0
    
    peaks_violation_count = 0
    max_peaks_len = 0
    
    print(f"\nChecking {len(ds)} samples...")
    print(f" > SMILES Limit: {smiles_limit} tokens")
    print(f" > Peaks Limit:  {peaks_limit} peaks")
    
    # Iterate through raw dataframe for speed (avoid full tensor conversion)
    for idx, row in tqdm(ds.df.iterrows(), total=len(ds.df)):
        # Check SMILES
        smiles = str(row['smiles'])
        # Encode without truncation to find real length
        tokens = tokenizer.encode(smiles, add_special_tokens=True)
        t_len = len(tokens)
        
        if t_len > max_smiles_len: max_smiles_len = t_len
        if t_len > smiles_limit: smiles_violation_count += 1
        
        # Check Peaks (Count commas + 1)
        try:
            mzs_str = str(row['mzs'])
            p_len = mzs_str.count(',') + 1
            
            if p_len > max_peaks_len: max_peaks_len = p_len
            if p_len > peaks_limit: peaks_violation_count += 1
        except:
            pass

    print("\n" + "="*40)
    print(" RESULTS")
    print("="*40)
    
    print(f"Max SMILES Length Found: {max_smiles_len} tokens")
    print(f"SMILES Truncated: {smiles_violation_count} / {len(ds)} ({smiles_violation_count/len(ds)*100:.2f}%)")
    if smiles_violation_count > 0:
        print(" -> NOTE: These will be safely truncated by the tokenizer.")
    else:
        print(" -> SAFE: All SMILES fit within the limit.")
        
    print("-" * 40)
    
    print(f"Max Peaks Found: {max_peaks_len}")
    print(f"Spectra Truncated: {peaks_violation_count} / {len(ds)} ({peaks_violation_count/len(ds)*100:.2f}%)")
    if peaks_violation_count > 0:
        print(" -> NOTE: These will be truncated to the top-k highest intensity peaks.")
    
    print("="*40)