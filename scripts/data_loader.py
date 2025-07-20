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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image as PIL
        image = Image.open(row['image_path']).convert('RGB')

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

        label = torch.tensor(row['label'], dtype=torch.long)

        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }
    
def get_transforms(image_size=256, is_training=False):
    if is_training:
        return A.Compose([
            A.Rotate(
                limit=(11),  # Degrees
                p=0.8,
                border_mode=0  # cv2.BORDER_CONSTANT
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