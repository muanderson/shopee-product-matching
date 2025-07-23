# Shopee/engine.py

import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import ArcFaceLoss
from sklearn.metrics import pairwise_distances, f1_score
import numpy as np
import os

def recall_at_k(embeddings, labels, k=1):
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    dists = pairwise_distances(embeddings, embeddings, metric='cosine')
    indices = np.argsort(dists, axis=1)[:, 1:k+1]

    correct = 0
    for i, neighbors in enumerate(indices):
        if labels[i] in labels[neighbors]:
            correct += 1
    return correct / len(labels)

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            embeddings = model(images, input_ids, attention_mask)
            loss = criterion(embeddings, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            embeddings = model(images, input_ids, attention_mask)
            embeddings_list.append(embeddings)
            labels_list.append(labels)

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    r1 = recall_at_k(embeddings, labels, k=1)
    r5 = recall_at_k(embeddings, labels, k=5)

    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    dists = pairwise_distances(embeddings_np, embeddings_np, metric='cosine')
    np.fill_diagonal(dists, np.inf)
    nearest_idx = dists.argmin(axis=1)
    pred_labels = labels_np[nearest_idx]

    mean_f1 = f1_score(labels_np, pred_labels, average='macro', zero_division=0)

    return {'recall@1': r1, 'recall@5': r5, 'mean_f1': mean_f1}

def train_model(model, train_loader, val_loader, config, num_classes, embedding_size):
    device = torch.device(config['device'])
    model.to(device)

    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embedding_size,
        margin=28.6,
        scale=64.0
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler(enabled=config['use_amp'])

    best_recall = 0
    best_r1 = 0
    best_r5 = 0
    best_f1 = 0
    patience_counter = 0

    for epoch in range(config['epochs']):
        if epoch == config.get('unfreeze_epoch', -1):
            print(f"Unfreezing encoders at epoch {epoch}")
            model.unfreeze_backbones(
                unfreeze_bert_layers=config.get('unfreeze_bert_layers', 2),
                unfreeze_vit_layers=config.get('unfreeze_vit_layers', 2)
            )

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Recall@1: {val_metrics['recall@1']:.4f}")
        print(f"Val Recall@5: {val_metrics['recall@5']:.4f}")
        print(f"Val Mean F1: {val_metrics['mean_f1']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        current_metric = val_metrics['recall@5'] if epoch < 15 else val_metrics['recall@1']

        if current_metric > best_recall:
            best_recall = current_metric
            best_r1 = val_metrics['recall@1']
            best_r5 = val_metrics['recall@5']
            best_f1 = val_metrics['mean_f1']

            model_path = os.path.join(
                config['output_dir'],
                f"best_model_fold_{config['fold'] + 1}.pt"
            )
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{config['patience']}")

        if patience_counter >= config['patience']:
            print("Early stopping triggered.")
            break

        scheduler.step()

    return best_r1, best_r5, best_f1
