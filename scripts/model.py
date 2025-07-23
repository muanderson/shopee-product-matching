# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import BertModel, BertTokenizer

class Shopeetransformer(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', embed_dim=512):
        super().__init__()

        # Vision Transformer
        weights = ViT_B_16_Weights.DEFAULT
        self.image_encoder = vit_b_16(weights=weights)
        self.image_encoder.heads = nn.Identity()
        self.image_proj = nn.Linear(768, embed_dim)

        # BERT
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        # Final projection
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

        # Initially freeze both encoders
        self.freeze_backbones()

    def freeze_backbones(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self, unfreeze_bert_layers=2, unfreeze_vit_layers=2):
        # Unfreeze last N BERT encoder layers
        for name, param in self.text_encoder.named_parameters():
            if any(f'encoder.layer.{i}' in name for i in range(12 - unfreeze_bert_layers, 12)):
                param.requires_grad = True

        # Unfreeze last N ViT encoder layers
        for i in range(12 - unfreeze_vit_layers, 12):
            for param in self.image_encoder.encoder.layers[i].parameters():
                param.requires_grad = True

    def forward(self, image, input_ids, attention_mask):
        # Image encoding
        image_feat = self.image_encoder(image)  # [B, 768]
        image_feat = self.image_proj(image_feat)  # [B, embed_dim]

        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token from the last_hidden_state for a better sentence embedding
        text_feat = text_outputs.last_hidden_state[:, 0]
        # --------------------
        
        text_feat = self.text_proj(text_feat)  # [B, embed_dim]

        # Concatenate & fuse
        fused = torch.cat([image_feat, text_feat], dim=1)  # [B, 2*embed_dim]
        fused = self.fusion_proj(fused)  # [B, embed_dim]

        # L2 normalisation for metric learning
        fused = F.normalize(fused, p=2, dim=1)

        return fused
