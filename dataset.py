
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
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# --- HELPER FUNCTIONS (Must be outside class for Parallel Processing) ---

def canonicalize_smiles(s):
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None: return s
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return s

def process_candidate_item(args):
    """
    Worker function to process a single dictionary item in parallel.
    args: (key, value_list, set_of_valid_keys)
    """
    k, v_list, relevant_smiles = args
    
    # 1. Canonicalize the Key (Query)
    can_k = canonicalize_smiles(k)
    
    # 2. Check if this query is actually in our validation set
    #    (If not, skip the heavy work of processing its 256 candidates)
    if can_k in relevant_smiles:
        # 3. Canonicalize the Values (Candidates)
        can_v = [canonicalize_smiles(c) for c in v_list]
        return (can_k, can_v)
    
    return None

# -----------------------------------------------------------------------

class MassSpecGymTrainDataset(Dataset):
    def __init__(self, tokenizer, split, max_len=300, retrieval_mode=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.retrieval_mode = retrieval_mode
        
        if isinstance(split, str):
            self.split_names = [split]
        else:
            self.split_names = split
            
        split_sig = "_".join(self.split_names)
        print(f"\n[Data] Initializing MassSpecGym for split(s): {self.split_names} (Retrieval Mode: {retrieval_mode})")

        # --- CACHE CHECK ---
        cache_filename = f"cached_massspecgym_{split_sig}_{retrieval_mode}.pkl"
        
        if os.path.exists(cache_filename):
            print(f"   -> 🚀 Fast-loading from cache: {cache_filename}")
            with open(cache_filename, 'rb') as f:
                data = pickle.load(f)
                self.df = data['df']
                self.candidates_map = data.get('candidates_map', {})
            print(f"   -> ✅ Loaded {len(self.df)} samples.")
            return

        # --- LOAD FROM SCRATCH ---
        print("   -> ⏳ Cache not found. Loading from source...")
        
        # 1. Load TSV
        try:
            tsv_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.FILENAME_TSV, repo_type="dataset")
        except:
            tsv_path = config.FILENAME_TSV
            
        df = pd.read_csv(tsv_path, sep="\t")
        
        # Filter splits
        self.df = df[df['fold'].isin(self.split_names)].reset_index(drop=True)
        self.df = self.df.dropna(subset=['smiles'])
        
        print(f"   -> Canonicalizing {len(self.df)} TSV SMILES...")
        # (This part is usually fast enough, but could be parallelized if needed)
        self.df['smiles'] = self.df['smiles'].apply(canonicalize_smiles)
        
        # 2. Load Candidates (ONLY for Validation)
        self.candidates_map = {}
        if self.retrieval_mode:
            print(f"   -> Loading Candidates JSON...")
            try:
                json_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.FILENAME_CANDIDATES, repo_type="dataset")
            except:
                json_path = config.FILENAME_CANDIDATES
            
            with open(json_path, 'r') as f:
                raw_map = json.load(f)
            
            # --- PARALLEL PROCESSING START ---
            print(f"   -> ⚡ Canonicalizing Candidate Map (Parallel Processing)...")
            
            # Create a set for O(1) lookups
            relevant_smiles = set(self.df['smiles'].values)
            
            # Prepare arguments for workers
            # We only send items from the map, coupled with the valid keys set
            tasks = [(k, v, relevant_smiles) for k, v in raw_map.items()]
            
            # Use all available CPU cores
            with ProcessPoolExecutor() as executor:
                # Wrap in tqdm for a progress bar
                results = list(tqdm(
                    executor.map(process_candidate_item, tasks), 
                    total=len(tasks),
                    desc="Processing Candidates"
                ))
            
            # Filter out None results (skipped items) and build dict
            for res in results:
                if res is not None:
                    k, v = res
                    self.candidates_map[k] = v
            # --- PARALLEL PROCESSING END ---

            valid_indices = [i for i, row in self.df.iterrows() if row['smiles'] in self.candidates_map]
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)
            print(f"   -> Filtered to {len(self.df)} queries with valid candidates.")

        # --- SAVE CACHE ---
        print(f"   -> 💾 Saving cache to {cache_filename}...")
        with open(cache_filename, 'wb') as f:
            pickle.dump({'df': self.df, 'candidates_map': self.candidates_map}, f)

        print(f"✅ Loaded {len(self.df)} spectra.")

    def __len__(self):
        return len(self.df)
    
    def _process_spectrum(self, mzs_str, ints_str):
        # ... (Same as before) ...
        try:
            mzs = np.array([float(x) for x in mzs_str.split(',')])
            intensities = np.array([float(x) for x in ints_str.split(',')])
        except:
            return torch.zeros(self.max_len, 2), torch.zeros(self.max_len, dtype=torch.bool)

        root_ints = np.sqrt(intensities)
        max_int = root_ints.max() if len(root_ints) > 0 else 1.0
        if max_int < 1e-9: max_int = 1.0
        norm_ints = root_ints / max_int
        
        log_mzs = np.log(mzs + 1.0)
        
        processed = list(zip(log_mzs, norm_ints))
        
        if len(processed) > self.max_len:
            processed.sort(key=lambda x: x[1], reverse=True)
            processed = processed[:self.max_len]
        
        processed.sort(key=lambda x: x[0])
        
        peak_sequence = torch.zeros(self.max_len, 2, dtype=torch.float32)
        peak_mask = torch.zeros(self.max_len, dtype=torch.bool)
        
        if len(processed) > 0:
            peak_sequence[:len(processed)] = torch.tensor(processed)
            peak_mask[:len(processed)] = True
            
        return peak_sequence, peak_mask

    def __getitem__(self, idx):
        # ... (Same as before) ...
        row = self.df.iloc[idx]
        smiles = row['smiles']
        
        peak_seq, peak_mask = self._process_spectrum(row['mzs'], row['intensities'])
        
        item = {
            'peak_sequence': peak_seq,
            'peak_mask': peak_mask,
        }

        if self.retrieval_mode:
            item['smiles'] = smiles
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
    from tqdm import tqdm
    import config
    import random

    print("--- RUNNING DATASET SAFETY CHECK ---")
    
    # 1. Initialize Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER['model_name'])
    
    # 2. Load Dataset (Validation Mode to check BOTH TSV and JSON)
    print("\n[Setup] Initializing Dataset in RETRIEVAL MODE...")
    ds = MassSpecGymTrainDataset(tokenizer, split='val', retrieval_mode=True)
    
    # 3. Check Variables (Spectra & SMILES limits)
    smiles_limit = config.TEXT_ENCODER['max_length']
    peaks_limit = config.MAX_PEAK_SEQ_LEN
    
    smiles_violation_count = 0
    max_smiles_len = 0
    
    peaks_violation_count = 0
    max_peaks_len = 0
    
    print(f"\n[Check 1] Analyzing {len(ds)} spectra samples (TSV File)...")
    print(f" > SMILES Limit: {smiles_limit} tokens")
    print(f" > Peaks Limit:  {peaks_limit} peaks")
    
    for idx, row in tqdm(ds.df.iterrows(), total=len(ds.df)):
        # Check SMILES
        smiles = str(row['smiles'])
        tokens = tokenizer.encode(smiles, add_special_tokens=True)
        t_len = len(tokens)
        
        if t_len > max_smiles_len: max_smiles_len = t_len
        if t_len > smiles_limit: smiles_violation_count += 1
        
        # Check Peaks
        try:
            mzs_str = str(row['mzs'])
            p_len = mzs_str.count(',') + 1
            
            if p_len > max_peaks_len: max_peaks_len = p_len
            if p_len > peaks_limit: peaks_violation_count += 1
        except:
            pass

    # 4. Check Candidates (JSON File)
    print("\n[Check 2] Analyzing Candidates (JSON File)...")
    if hasattr(ds, 'candidates_map') and len(ds.candidates_map) > 0:
        print(f" > Successfully loaded candidate map with {len(ds.candidates_map)} keys.")
        
        # Pick a random sample to verify
        sample_idx = random.randint(0, len(ds) - 1)
        item = ds[sample_idx]
        
        gt_smiles = item['smiles']
        candidates = item['candidates_smiles']
        
        print(f" > Sample verification (Index {sample_idx}):")
        print(f"   - Ground Truth: {gt_smiles[:30]}...")
        print(f"   - Candidate Count: {len(candidates)}")
        
        # Crucial Check: Is Ground Truth in Candidates?
        if gt_smiles in candidates:
            print("   - ✅ SAFETY CHECK PASSED: Ground truth is inside candidate list.")
        else:
            print("   - ❌ WARNING: Ground truth NOT found in candidate list!")
            
    else:
        print(" > ❌ WARNING: Candidate map is empty or not loaded!")

    print("\n" + "="*40)
    print(" FINAL RESULTS")
    print("="*40)
    
    print(f"Max SMILES Length: {max_smiles_len} (Truncated: {smiles_violation_count}/{len(ds)})")
    print(f"Max Peaks Length:  {max_peaks_len} (Truncated: {peaks_violation_count}/{len(ds)})")
    print("="*40)