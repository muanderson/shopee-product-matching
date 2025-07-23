# Shopee/data_loader.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer
from PIL import Image

class ShopeeDataset(Dataset):
    def __init__(self, dataframe, transform=None, tokenizer_name='bert-base-uncased', max_length=128):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'image_path', 'title', 'label' columns
            transform (callable, optional): Image transforms
            tokenizer_name (str): Pretrained tokenizer name
            max_length (int): Max token length for text
        """
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        # Check if labels are available
        self.has_labels = 'label' in self.data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image as PIL
        image = Image.open(row['image_path']).convert('RGB')

        width, height = image.size

        if width or height != 224:
            image = image.resize((224, 224), resample=Image.LANCZOS)

        # Convert PIL to numpy for Albumentations
        image_np = np.array(image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image = augmented['image']

        # Tokenize text
        text = row['title']
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Conditionally get the label
        if self.has_labels:
            label = torch.tensor(row['label'], dtype=torch.long)
        else:
            # For test data, provide a dummy placeholder
            label = torch.tensor(-1, dtype=torch.long)
            
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }
    
def get_transforms(image_size=224, is_training=False):
    if is_training:
        return A.Compose([
            A.Rotate(
                limit=(11),
                p=0.8,
                border_mode=0 
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
