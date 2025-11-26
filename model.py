# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import config
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(1, embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        # x: [Batch, Seq, 1]
        x_proj = (2 * np.pi * x) @ self.B 
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MSEncoder(nn.Module):
    def __init__(self, encoder_config, embedding_dim):
        super().__init__()
        
        d_model = encoder_config['d_model'] 
        fourier_dim = encoder_config['fourier_dim'] 
        
        # 1. m/z Pathway
        self.mz_fourier = GaussianFourierProjection(fourier_dim, scale=30.0)
        self.mz_mlp = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Instead of 1 -> 768, we do 1 -> 64 -> 768
        # This prevents the model from over-interpreting the scalar noise.
        self.int_mlp = nn.Sequential(
            nn.Linear(1, 64),         # Bottleneck
            nn.GELU(),
            nn.Linear(64, d_model),   # Expand
            nn.LayerNorm(d_model)     # Stabilize
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # --- 4. Transformer ---
        # [FIXED] Initialize with random noise, not zeros
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=encoder_config['nhead'],
            dim_feedforward=d_model * 4,
            dropout=encoder_config['dropout'],
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=encoder_config['n_layers']
        )
        
        self.projection = nn.Sequential(
            nn.Linear(d_model, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(encoder_config['dropout']),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def forward(self, x, mask):
        # x: [B, S, 2] -> (log_mz, sqrt_int)
        mz_val = x[:, :, 0:1]
        int_val = x[:, :, 1:2]
        
        # 1. Encode m/z
        mz_emb = self.mz_fourier(mz_val) 
        mz_emb = self.mz_mlp(mz_emb)     
        
        # 2. Encode intensity [FIXED]
        int_emb = self.int_mlp(int_val)
        
        # 3. Concatenate & Fuse
        cat_feat = torch.cat([mz_emb, int_emb], dim=-1) 
        x_emb = self.fusion(cat_feat) 
        
        # 4. Append CLS
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        
        # 5. Masking
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat((cls_mask, mask), dim=1)
        transformer_mask = ~full_mask
        
        # 6. Transformer
        out = self.transformer_encoder(x_emb, src_key_padding_mask=transformer_mask)
        
        # 7. Output
        cls_output = out[:, 0, :]
        projected = self.projection(cls_output)
        return F.normalize(projected, p=2, dim=1, eps=1e-6)

class TextEncoder(nn.Module):
    def __init__(self, model_name, embedding_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, use_safetensors=True)
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.LORA['r'],
            lora_alpha=config.LORA['lora_alpha'],
            lora_dropout=config.LORA['lora_dropout'],
            target_modules=config.LORA['target_modules']
        )
        self.bert = get_peft_model(self.bert, lora_config)
        
        bert_dim = self.bert.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        mean_pooled_embedding = sum_embeddings / (sum_mask + 1e-9)
        
        projected = self.projection(mean_pooled_embedding)
        return F.normalize(projected, p=2, dim=1, eps=1e-6)

class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ms_encoder = MSEncoder(config.MS_ENCODER, config.EMBEDDING_DIM)
        self.text_encoder = TextEncoder(config.TEXT_ENCODER['model_name'], config.EMBEDDING_DIM)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.TEMPERATURE))

    def forward(self, peak_sequence, peak_mask, input_ids, attention_mask):
        spec_embeds = self.ms_encoder(peak_sequence, peak_mask)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        
        batch_size = spec_embeds.shape[0]
        logits_per_spec = (spec_embeds @ text_embeds.T) * self.logit_scale.exp()
        logits_per_text = logits_per_spec.T
        labels = torch.arange(batch_size, device=config.DEVICE, dtype=torch.long)
        
        loss_spec = F.cross_entropy(logits_per_spec, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        return (loss_spec + loss_text) / 2