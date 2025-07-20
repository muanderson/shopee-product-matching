# Shopee/train.py

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data_loader import Shopeedataset, get_transforms
from model import Shopeetransformer
from engine import train_model

def seed_everything(seed=42):
    """
    Set random seeds for reproducibility across numpy, torch, and CUDA.

    Args:
        seed (int): Random seed value, defaults to 42
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main training pipeline orchestrating the entire training process.
    Handles data loading, preprocessing, model training, and cross-validation.
    """

    # Set random seed
    seed_everything()

    # Configuration dictionary from command line arguments
    config = {
        # Paths
        'output_dir': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching',
        'image_dir': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching\data\train_images',
        'tabular_data_path': r'C:\Users\Matthew\Documents\PhD\shopee-product-matching\data\train.csv',

        # Training parameters
        'image_size': 256,
        'batch_size': 32,
        'learning_rate': 0.001,
        'min_lr': 1e-6,
        'weight_decay': weight_decay,
        't0': t0,
        't_mult': t_mult,
        'epochs': 100,
        'patience': 100,
        'device': torch.device(cuda:0 if torch.cuda.is_available() else "cpu"),
        'n_splits': 5,
    }

    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)

    # Load and preprocess clinical data
    product_data = pd.read_csv(config['tabular_data_path'])

    # Image directory
    image_dir = config['image_dir']

    # Find images linked to multiple labels
    ambiguous_images = product_data.groupby('image')['label_group'].nunique()
    ambiguous_images = ambiguous_images[ambiguous_images > 1].index

    # Find titles linked to multiple labels
    ambiguous_titles = product_data.groupby('title')['label_group'].nunique()
    ambiguous_titles = ambiguous_titles[ambiguous_titles > 1].index

    # Flag ambiguous samples
    product_data['ambiguous'] = product_data['image'].isin(ambiguous_images) | product_data['title'].isin(ambiguous_titles)

    # Separate ambiguous and clean data
    product_ambiguous = product_data[product_data['ambiguous']]
    product_clean = product_data[~product_data['ambiguous']]

    # Create mapping from label_group to label_id
    label_map = {label: idx for idx, label in enumerate(product_clean['label_group'].unique())}
    product_clean['label'] = product_clean['label_group'].map(label_map)

    # Verify image paths
    product_clean['image_path'] = product_clean['image'].apply(lambda x: os.path.join(image_dir, x))

    # Setup cross-validation
    kfold = KFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    fold_results = []

    X = product_clean.index  # or even just np.arange(len(product_clean))
    y = product_clean['label']

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        config['fold'] = fold
        print(f'\nTraining Fold {fold + 1}/{config["n_splits"]}')

        # Split data for current fold
        train_data = product_clean.iloc[train_idx].reset_index(drop=True)
        val_data = product_clean.iloc[val_idx].reset_index(drop=True)

        # Create datasets and dataloaders
        train_dataset = Shopeedataset(
            train_data,
            transform=get_transforms(config['image_size'], is_training=True)
        )

        val_dataset = Shopeedataset(
            val_data,
            transform=get_transforms(config['image_size'], is_training=False)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Initialize and train model
        model = Shopeetransformer.to(config['device'])
        best_mae, best_mse, best_r2 = train_model(model, train_loader, val_loader, config)

        print(f'Fold {fold + 1}, Best MAE: {best_mae:.4f}, Best MSE: {best_mse:.4f}, Best R²: {best_r2:.4f}')
        fold_results.append((best_mae, best_mse, best_r2))

    # Print cross-validation results
    if fold_results:
        avg_mae = sum(result[0] for result in fold_results) / len(fold_results)
        avg_mse = sum(result[1] for result in fold_results) / len(fold_results)
        avg_r2 = sum(result[2] for result in fold_results) / len(fold_results)
        print('\n=== Cross-Validation Results ===')
        for i, (mae, mse, r2) in enumerate(fold_results, 1):
            print(f'Fold {i}: MAE={mae:.4f}, MSE={mse:.4f}, R²={r2:.4f}')
        print(f'Average MAE: {avg_mae:.4f}, Average MSE: {avg_mse:.4f}, Average R²: {avg_r2:.4f}')
    else:
        print("No folds were completed successfully.")

if __name__ == "__main__":
    main()