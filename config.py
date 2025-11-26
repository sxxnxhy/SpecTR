
# Device
DEVICE = 'cuda:0'


MAX_PEAK_SEQ_LEN = 300    # 스펙트럼 당 최대 피크 (토큰) 시퀀스 길이

# Model Architecture
EMBEDDING_DIM = 768       # 공유 임베딩 차원 (BERT와 일치)


MS_ENCODER = {
    # Gaussian Fourier Projection Dimension (Internal)
    # This projects scalar m/z -> vector of size 'fourier_dim'
    'fourier_dim': 256,
    # (m/z, intensity) 2D 입력을 d_model로 임베딩
    'd_model': 256, 
    'nhead': 8,           # (d_model % nhead == 0)
    'n_layers': 6,        # 트랜스포머 레이어 수
    'dropout': 0.1
}

# loss temperature
TEMPERATURE = 0.07

# Text Encoder
TEXT_ENCODER = {
    'model_name': 'seyonec/PubChem10M_SMILES_BPE_450k',
    'max_length': 222,  
    'freeze_bert': False
}

LORA = {
    'r': 16, 
    'lora_alpha' : 32, 
    'lora_dropout': 0.1,
    'target_modules': ["query", "key", "value"] # Apply to attention layers
}

# Training
BATCH_SIZE = 128
NUM_EPOCHS = 500 
WEIGHT_DECAY = 1e-2 

# Learning Rates 
LR_LORA = 1e-4
LR_ENCODER = 1e-4 


# MassSpecGym files (automatically downloaded)
REPO_ID = "roman-bushuiev/MassSpecGym"
FILENAME_TSV = "data/MassSpecGym.tsv"
FILENAME_CANDIDATES = "molecules/MassSpecGym_retrieval_candidates_mass.json"
MASSSPECGYM_LOG_DIR = './logs'
MASSSPECGYM_CKPT_DIR = './models'