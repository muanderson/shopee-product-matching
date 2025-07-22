# Shopee/train.py

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from data_loader import ShopeeDataset, get_transforms
from model import Shopeetransformer

from engine import train_model

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    seed_everything()

    config = {
        # Directories
        'output_dir': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching\models',
        'image_dir': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching\data\train_images',
        'tabular_data_path': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching\data\train.csv',

        # Variables
        'image_size': 224,
        'batch_size': 64,
        'learning_rate': 5e-4,
        'epochs': 100,
        'patience': 10,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'n_splits': 5,
        'use_amp': True,

        # Staged unfreezing config
        'unfreeze_epoch': 7,
        'unfreeze_bert_layers': 2,
        'unfreeze_vit_layers': 2,
        
        # Embedding dimension for the model and ArcFace loss
        'embedding_size': 512,
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    product_data = pd.read_csv(config['tabular_data_path'])
    image_dir = config['image_dir']

    ambiguous_images = product_data.groupby('image')['label_group'].nunique()
    ambiguous_images = ambiguous_images[ambiguous_images > 1].index

    ambiguous_titles = product_data.groupby('title')['label_group'].nunique()
    ambiguous_titles = ambiguous_titles[ambiguous_titles > 1].index

    product_data['ambiguous'] = product_data['image'].isin(ambiguous_images) | product_data['title'].isin(ambiguous_titles)
    product_clean = product_data[~product_data['ambiguous']].copy()

    unique_label_groups = product_clean['label_group'].unique()
    label_map = {label: idx for idx, label in enumerate(unique_label_groups)}
    product_clean['label'] = product_clean['label_group'].map(label_map)
    product_clean['image_path'] = product_clean['image'].apply(lambda x: os.path.join(image_dir, x))

    num_classes = product_clean['label'].nunique()

    kfold = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_results = []

    X = product_clean.index
    y = product_clean['label']

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        config['fold'] = fold
        print(f'\nTraining Fold {fold + 1}/{config["n_splits"]}')

        train_data = product_clean.iloc[train_idx].reset_index(drop=True)
        val_data = product_clean.iloc[val_idx].reset_index(drop=True)

        train_dataset = ShopeeDataset(
            train_data,
            transform=get_transforms(config['image_size'], is_training=True)
        )
        val_dataset = ShopeeDataset(
            val_data,
            transform=get_transforms(config['image_size'], is_training=False)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        # Pass the embedding size to the model
        model = Shopeetransformer(embed_dim=config['embedding_size'])
        model = model.to(config['device'])

        # --- UPDATED FUNCTION CALL ---
        best_r1, best_r5, best_f1 = train_model(
            model, 
            train_loader, 
            val_loader, 
            config, 
            num_classes, 
            config['embedding_size']
        )

        print(f'Fold {fold + 1}, Best Recall@1: {best_r1:.4f}, Best Recall@5: {best_r5:.4f}, Best MeanF1: {best_f1:.4f}')
        fold_results.append((best_r1, best_r5))

    if fold_results:
        avg_r1 = sum(r[0] for r in fold_results) / len(fold_results)
        avg_r5 = sum(r[1] for r in fold_results) / len(fold_results)
        print('\n=== Cross-Validation Results ===')
        for i, (r1, r5) in enumerate(fold_results, 1):
            print(f'Fold {i}: Recall@1={r1:.4f}, Recall@5={r5:.4f}')
        print(f'Average Recall@1: {avg_r1:.4f}, Average Recall@5: {avg_r5:.4f}')
    else:
        print("No folds were completed successfully.")

if __name__ == "__main__":
    main()