"""
Training script for MassBank CLIP-style Zero-Shot Retrieval (ZSR)
BUT using the MassSpecGym Training Data (Official Benchmark Split).
"""
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

import config
from model import CLIPModel 
from transformers import AutoTokenizer
from dataset import MassSpecGymTrainDataset

# Suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_massspecgym_loaders():
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER['model_name'])
    
    train_dataset = MassSpecGymTrainDataset(tokenizer, split='train')
    val_dataset = MassSpecGymTrainDataset(tokenizer, split='validation') # Use validation fold for checking
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader


@torch.no_grad()
def validate_simple(model, val_loader, device):
    """Simple R@1 check on the Validation Fold (Not the Test Fold)"""
    model.eval()
    all_spec = []
    all_text = []
    
    for batch in tqdm(val_loader, desc="Validating"):
        p_seq = batch['peak_sequence'].to(device)
        p_mask = batch['peak_mask'].to(device)
        i_ids = batch['input_ids'].to(device)
        a_mask = batch['attention_mask'].to(device)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            s_emb = model.ms_encoder(p_seq, p_mask)
            t_emb = model.text_encoder(i_ids, a_mask)
            
        all_spec.append(s_emb.cpu())
        all_text.append(t_emb.cpu())
        
    all_spec = torch.cat(all_spec, dim=0).float()
    all_text = torch.cat(all_text, dim=0).float()
    
    # Similarity
    sim = all_spec @ all_text.T
    labels = torch.arange(len(all_spec))
    
    # R@1
    top1 = sim.argmax(dim=1)
    acc = (top1 == labels).float().mean().item()
    return acc

def train_one_epoch(model, loader, optimizer, scaler, device):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def main():
    print("Training on MassSpecGym Official Data")

    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    Path(config.MASSSPECGYM_CKPT_DIR).mkdir(exist_ok=True)
    Path(config.MASSSPECGYM_LOG_DIR).mkdir(exist_ok=True)
    
    # Loaders
    train_loader, val_loader = prepare_massspecgym_loaders()
    
    # Model
    model = CLIPModel().to(device)
    
    # Optimizer (Same settings as before)
    lora_params = model.text_encoder.bert.parameters()
    enc_params = list(model.ms_encoder.parameters()) + \
                 list(model.text_encoder.projection.parameters()) + \
                 [model.logit_scale]
                 
    optimizer = optim.AdamW([
        {'params': lora_params, 'lr': config.LR_ENCODER},
        {'params': enc_params, 'lr': config.LR_ENCODER}
    ], weight_decay=config.WEIGHT_DECAY)
    
    scaler = GradScaler()
    writer = SummaryWriter(config.MASSSPECGYM_LOG_DIR)
    
    best_val_acc = 0.0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}")
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_acc = validate_simple(model, val_loader, device)
        
        print(f"Loss: {loss:.4f} | Val R@1 (In-Batch): {val_acc*100:.2f}%")
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Val/Acc", val_acc, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, f"{config.MASSSPECGYM_CKPT_DIR}/best_model.pt")
            print(f"✓ Saved Best Model")

if __name__ == '__main__':
    main()