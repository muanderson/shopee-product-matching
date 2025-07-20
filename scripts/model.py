# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import BertModel, BertTokenizer

class Shopeetransformer(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', image_out_dim=512, fusion_out_dim=512, num_classes=11014):
        super().__init__()

        # Pretrained ViT
        weights = ViT_B_16_Weights.DEFAULT
        self.image_encoder = vit_b_16(weights=weights)
        self.image_encoder.heads = nn.Identity()
        self.image_proj = nn.Linear(768, image_out_dim)

        # Pretrained BERT
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, image_out_dim)

        # Fusion + classifier
        self.classifier = nn.Sequential(
            nn.Linear(image_out_dim * 2, fusion_out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_out_dim, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # ViT feature extraction
        vit_outputs = self.image_encoder(image)
        # Use CLS token embedding (first token)
        image_feat = vit_outputs.last_hidden_state[:, 0]
        image_feat = self.image_proj(image_feat)

        # BERT feature extraction
        bert_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = bert_outputs.pooler_output
        text_feat = self.text_proj(text_feat)

        # Fuse
        fused = torch.cat([image_feat, text_feat], dim=1)

        # Classification logits
        logits = self.classifier(fused)
        return logits


