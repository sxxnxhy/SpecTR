import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
from pathlib import Path
import logging
import warnings
from torch.utils.data import DataLoader
import numpy as np
import random
from rdkit import RDLogger

# [ROBUSTNESS] Import Scheduler
from transformers import get_linear_schedule_with_warmup

import config
from model import CLIPModel 
from transformers import AutoTokenizer
from dataset import MassSpecGymTrainDataset

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*') 

# [ROBUSTNESS] Seed Everything for Reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def retrieval_hit_rate(scores, labels, top_k):
    if len(scores) <= top_k:
        return labels.sum().item() > 0
    top_k_indices = torch.topk(scores, k=top_k).indices
    hits = labels[top_k_indices].sum()
    return (hits > 0).float().item()

def prepare_loaders():
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER['model_name'])
    
    print(">>> Creating Training Loader (Split: train)...")
    train_dataset = MassSpecGymTrainDataset(
        tokenizer, 
        split='train', 
        retrieval_mode=False
    )    
    
    print(">>> Creating Validation Loader (Split: val)...")
    val_dataset = MassSpecGymTrainDataset(
        tokenizer, 
        split='val', 
        retrieval_mode=True
    )    
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True # [ROBUSTNESS] Drop incomplete batch to stabilize BatchNorm/Training
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=None 
    )
    
    return train_loader, val_loader, tokenizer

@torch.no_grad()
def validate_retrieval(model, val_loader, tokenizer, device):
    model.eval()
    
    metrics = {'hit_rate@1': [], 'hit_rate@5': [], 'hit_rate@20': []}
    
    for batch in tqdm(val_loader, desc="Retrieval Eval"):
        p_seq = batch['peak_sequence'].to(device)
        p_mask = batch['peak_mask'].to(device)
        true_smiles = batch['smiles'][0] 
        candidates = batch['candidates_smiles'] 
        if isinstance(candidates[0], list): candidates = candidates[0] 
        
        # OOM Safety Check: If candidates > 500, maybe handle differently?
        # For MassSpecGym (256), it fits on most 16GB+ GPUs.
        
        with autocast(device_type='cuda', dtype=torch.float16):
            query_emb = model.ms_encoder(p_seq, p_mask) 
            
        cand_tokens = tokenizer(
            candidates,
            max_length=config.TEXT_ENCODER['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            cand_embs = model.text_encoder(
                cand_tokens['input_ids'], 
                cand_tokens['attention_mask']
            ) 
            
        scores = (query_emb @ cand_embs.T).squeeze(0) 
        labels = torch.tensor([1 if c == true_smiles else 0 for c in candidates], device=device)
        
        for k in [1, 5, 20]:
            h = retrieval_hit_rate(scores, labels, top_k=k)
            metrics[f'hit_rate@{k}'].append(h)

    results = {k: np.mean(v) for k, v in metrics.items()}
    return results

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        p_seq = batch['peak_sequence'].to(device)
        p_mask = batch['peak_mask'].to(device)
        i_ids = batch['input_ids'].to(device)
        a_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            loss = model(p_seq, p_mask, i_ids, a_mask)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # [ROBUSTNESS] Norm clip 1.0 is standard for BERT
        scaler.step(optimizer)
        scaler.update()
        
        # [ROBUSTNESS] Step the scheduler
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def main():
    print("Training on MassSpecGym (Official Train Split)")
    
    # [ROBUSTNESS] Set Seed
    seed_everything(config.SEED)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    Path(config.MASSSPECGYM_CKPT_DIR).mkdir(exist_ok=True)
    Path(config.MASSSPECGYM_LOG_DIR).mkdir(exist_ok=True)
    
    train_loader, val_loader, tokenizer = prepare_loaders()
    
    model = CLIPModel().to(device)
    
    lora_params = model.text_encoder.bert.parameters()
    enc_params = list(model.ms_encoder.parameters()) + \
                 list(model.text_encoder.projection.parameters()) + \
                 [model.logit_scale]
                 
    optimizer = optim.AdamW([
        {'params': lora_params, 'lr': config.LR_LORA}, 
        {'params': enc_params, 'lr': config.LR_ENCODER}
    ], weight_decay=config.WEIGHT_DECAY)
    
    # [ROBUSTNESS] Scheduler Setup
    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_STEPS) # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()
    writer = SummaryWriter(config.MASSSPECGYM_LOG_DIR)
    
    best_hr1 = 0.0
    start_epoch = 1
    
    # [ROBUSTNESS] Check for Resume
    last_ckpt_path = f"{config.MASSSPECGYM_CKPT_DIR}/last.pt"
    if os.path.exists(last_ckpt_path):
        print(f"🔄 Resuming from checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_hr1 = ckpt['best_hr1']
        print(f"   -> Resuming at Epoch {start_epoch}, Best R@1 so far: {best_hr1:.4f}")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        
        print("Validating retrieval...")
        val_metrics = validate_retrieval(model, val_loader, tokenizer, device)
        
        print(f"Loss: {loss:.4f}")
        print(f"R@1: {val_metrics['hit_rate@1']:.4f} | R@5: {val_metrics['hit_rate@5']:.4f}")
        
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Val/R@1", val_metrics['hit_rate@1'], epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        
        # Save Best Model
        if val_metrics['hit_rate@1'] > best_hr1:
            best_hr1 = val_metrics['hit_rate@1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                # Note: We usually don't need optimizer state for inference, 
                # but nice to have if we want to fine-tune the best model later.
            }, f"{config.MASSSPECGYM_CKPT_DIR}/best_model.pt")
            print(f"✓ Saved Best Model (R@1: {best_hr1:.4f})")
            
        # [ROBUSTNESS] Save "Last" Model for crash recovery
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_hr1': best_hr1
        }, last_ckpt_path)

if __name__ == '__main__':
    main()