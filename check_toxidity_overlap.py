import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from rdkit import Chem
from tqdm import tqdm
import deepchem as dc
import config  # config.py의 설정을 그대로 가져와서 씁니다

def canonicalize(smiles):
    """SMILES 표준화 (중복 제거용)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None

def load_tox_db(dataset_name="Tox21"):
    """DeepChem에서 독성 데이터 로드"""
    print(f"Loading {dataset_name} from DeepChem...")
    try:
        tasks, datasets, transformers = getattr(dc.molnet, f"load_{dataset_name.lower()}")()
        train, valid, test = datasets
        all_smiles = []
        for d in [train, valid, test]:
            all_smiles.extend(d.ids)  # DeepChem은 ids에 SMILES가 들어있음
        return set(all_smiles)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return set()

def main():
    print("="*60)
    print("☣️  Checking Toxicity Overlap in MassSpecGym")
    print("="*60)

    # 1. MassSpecGym 데이터 자동 다운로드 & 로드
    print(f"[1/3] Downloading MassSpecGym data from {config.REPO_ID}...")
    try:
        tsv_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.FILENAME_TSV, repo_type="dataset")
        print(f"  - Path: {tsv_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        return

    df = pd.read_csv(tsv_path, sep="\t")
    print(f"  - Total Spectra: {len(df)}")
    
    # 2. MassSpecGym SMILES 추출 및 표준화
    raw_smiles = df['smiles'].dropna().unique()
    print(f"  - Unique Molecules (Raw): {len(raw_smiles)}")
    
    mass_gym_smiles = set()
    print("[2/3] Canonicalizing MassSpecGym SMILES...")
    for s in tqdm(raw_smiles):
        c_s = canonicalize(s)
        if c_s:
            mass_gym_smiles.add(c_s)
            
    print(f"  - Valid Canonical Molecules: {len(mass_gym_smiles)}")

    # 3. 독성 DB와 교집합 확인
    print("\n[3/3] Comparing with Toxicity Databases...")
    
    # Tox21
    tox21_smiles = load_tox_db("Tox21")
    tox21_canon = {canonicalize(s) for s in tqdm(tox21_smiles, desc="Canonicalizing Tox21") if s}
    
    # ClinTox
    clintox_smiles = load_tox_db("ClinTox")
    clintox_canon = {canonicalize(s) for s in tqdm(clintox_smiles, desc="Canonicalizing ClinTox") if s}
    
    # 교집합 계산
    overlap_tox21 = mass_gym_smiles.intersection(tox21_canon)
    overlap_clintox = mass_gym_smiles.intersection(clintox_canon)
    total_overlap = overlap_tox21.union(overlap_clintox)
    
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print(f"  - Overlap with Tox21: {len(overlap_tox21)} molecules")
    print(f"  - Overlap with ClinTox: {len(overlap_clintox)} molecules")
    print(f"  - TOTAL Unique Toxic Molecules: {len(total_overlap)}")
    print("="*60)

    if len(total_overlap) > 500:
        print("✅ Great! 충분한 독성 데이터가 있습니다. (Fine-tuning 가능)")
    elif len(total_overlap) > 100:
        print("⚠️ Okay. 실험해볼 만한 최소 수량입니다.")
    else:
        print("❌ Not enough. 데이터가 너무 적습니다.")

if __name__ == '__main__':
    main()