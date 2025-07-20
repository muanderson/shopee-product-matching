# Shopee/engine.py

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
import os # Added for os.path.join

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0

    for batch in train_loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Updated to new recommended API: torch.amp.autocast
        with torch.amp.autocast(device_type=device.type):
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Ensure validation also uses autocast if training does, for consistent behavior
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')

    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'f1': f1
    }

def train_model(model, train_loader, val_loader, config):
    """
    Train the model using the specified configuration.

    Args:
        model (nn.Module): The neural network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (dict): Training configuration parameters

    Returns:
        tuple: Best validation metrics (accuracy, F1 score)
    """
    # Convert device string to torch.device object
    device = torch.device(config['device'])
    model.to(device) # Move model to the selected device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0 = 10)
    
    # Corrected GradScaler initialization
    scaler = torch.amp.GradScaler(device=device)

    best_val_acc = float('-inf')
    best_val_f1 = float('-inf')
    
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Training
        # Pass the torch.device object
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Validation
        # Pass the torch.device object
        val_metrics = validate(model, val_loader, criterion, device)

        # Print progress
        print(f'Epoch {epoch+1}/{config["epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_metrics["loss"]:.4f}')
        print(f'Val Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'Val F1: {val_metrics["f1"]:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')

        # Update best metrics
        # There's a 'val_metrics['mse']' in your original code, but 'mse' isn't returned by `validate`.
        # I'm assuming you meant to use val_metrics['loss'] or had another metric in mind.
        # For now, I'll use val_metrics['loss'] for patience tracking.
        if val_metrics['loss'] < float('inf'): # Placeholder for a proper check, assumes lower loss is better
             # You might want to track accuracy or F1 for patience if that's your primary goal
            pass # No specific action here for loss patience without a target MSE

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            # Save the model weights
            model_path = os.path.join(
                config['output_dir'],
                f"best_model_fold_{config['fold'] + 1}.pt"
            )
            torch.save(model.state_dict(), model_path)
            patience_counter = 0 # Reset patience on F1 improvement
        else:
            patience_counter += 1
            print(f'No improvement in F1. Patience counter: {patience_counter}/{config["patience"]}')


        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            # Patience counter is reset only on F1 improvement as per your original logic.
            # If you want to reset on accuracy too, add 'patience_counter = 0' here.


        # Early stopping check
        if patience_counter >= config['patience']:
            print(f'Early stopping triggered after {epoch+1} epochs due to no improvement in F1 score.')
            break

        scheduler.step()

    return best_val_acc, best_val_f1