import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import time
import logging
from tqdm import tqdm
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Changed to import the whole config module for more robust access
import config
from model_train_eval.dataset import CrimeDataset
from model_train_eval.model import MultimodalCrimeDetector
from utils import setup_logger, get_logger, get_timestamp, find_files_in_directory, save_checkpoint, load_checkpoint, set_seed

import torch.nn.functional as F

main_logger = None

def get_label_from_filename(filename, label_mapping):
    if '_label_A' in filename:
        return label_mapping['_label_A']
    return label_mapping['_label_N']

def create_data_splits(all_video_sub_paths, data_split_ratios, data_splits_dir, random_seed):
    global main_logger
    split_filepath = os.path.join(data_splits_dir, 'data_splits.json')
    os.makedirs(data_splits_dir, exist_ok=True)

    if os.path.exists(split_filepath):
        main_logger.info(f"Loading existing data splits from {split_filepath}")
        with open(split_filepath, 'r') as f:
            splits = json.load(f)
        return splits['train'], splits['val'], splits['test']
    
    main_logger.info("Creating new stratified data splits...")
    
    video_paths_with_labels = []
    for path in all_video_sub_paths:
        label = get_label_from_filename(os.path.splitext(path)[0], config.LABEL_MAPPING) # Use config.LABEL_MAPPING
        video_paths_with_labels.append((path, label))

    paths = [item[0] for item in video_paths_with_labels]
    labels = [item[1] for item in video_paths_with_labels]

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels,
        test_size=(data_split_ratios['val'] + data_split_ratios['test']),
        stratify=labels,
        random_state=random_seed
    )

    test_size_for_temp = data_split_ratios['test'] / (data_split_ratios['val'] + data_split_ratios['test'])
    
    if data_split_ratios['val'] == 0:
        val_paths = []
        test_paths = train_val_paths
    elif data_split_ratios['test'] == 0:
        val_paths = train_val_paths
        test_paths = []
    else:
        val_paths, test_paths, _, _ = train_test_split(
            train_val_paths, train_val_labels,
            test_size=test_size_for_temp,
            stratify=train_val_labels,
            random_state=random_seed
        )

    splits = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }

    with open(split_filepath, 'w') as f:
        json.dump(splits, f, indent=4)
    
    main_logger.info(f"Data splits created and saved to {split_filepath}")
    main_logger.info(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}, Test samples: {len(test_paths)}")
    
    return train_paths, val_paths, test_paths


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, log_interval):
    global main_logger
    model.train()
    total_loss = 0
    predictions, true_labels = [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False, file=sys.stdout)
    for batch_idx, batch in enumerate(pbar):
        text_features = batch['text_features'].to(device)
        audio_features = batch['audio_features'].to(device)
        video_features = batch['video_features'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(text_features, video_features, audio_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix(loss=loss.item())
    
    epoch_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    main_logger.info(f"Epoch {epoch} Training - Loss: {epoch_loss:.4f}, Acc: {accuracy*100:.2f}%")
    return epoch_loss, accuracy

def evaluate_model(model, data_loader, criterion, device, phase="Validation"):
    global main_logger
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    pbar = tqdm(data_loader, desc=f"{phase} Evaluation", leave=False, file=sys.stdout)

    with torch.no_grad():
        for batch in pbar:
            text_features = batch['text_features'].to(device)
            audio_features = batch['audio_features'].to(device)
            video_features = batch['video_features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(text_features, video_features, audio_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    auc = 0.0
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        main_logger.warning(f"AUC not calculated for {phase}: Only one class present in labels.")

    main_logger.info(f"{phase} Metrics - Loss: {avg_loss:.4f}, Acc: {accuracy*100:.2f}%, "
                     f"Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    return avg_loss, accuracy, precision, recall, f1, auc

def main():
    global main_logger
    log_dir = config.OUTPUT_DIRS['logs'] # Use config.OUTPUT_DIRS
    os.makedirs(log_dir, exist_ok=True)
    timestamp = get_timestamp()
    
    main_logger = setup_logger('TrainEvalRunner', os.path.join(log_dir, f'train_eval_run_{timestamp}.log'))
    main_logger.info("STARTING MULTIMODAL CRIME DETECTION TRAINING AND EVALUATION")
    main_logger.info(f"Log file: {os.path.join(log_dir, f'train_eval_run_{timestamp}.log')}")
    main_logger.info(f"Running on device: {config.DEVICE}") # Use config.DEVICE
    set_seed(config.RANDOM_SEED) # Use config.RANDOM_SEED

    main_logger.info("Collecting all video paths from raw_videos directory...")
    all_video_sub_paths = find_files_in_directory(config.BASE_PATHS['raw_videos'], ['.mp4']) # Use config.BASE_PATHS
    if not all_video_sub_paths:
        main_logger.error(f"No video files found in {config.BASE_PATHS['raw_videos']}. Exiting.") # Use config.BASE_PATHS
        return

    train_video_paths, val_video_paths, test_video_paths = create_data_splits(
        all_video_sub_paths, config.DATA_SPLIT_RATIOS, config.BASE_PATHS['data_splits'], config.RANDOM_SEED # Use config.DATA_SPLIT_RATIOS, config.BASE_PATHS, config.RANDOM_SEED
    )

    main_logger.info("Initializing datasets and data loaders...")
    train_dataset = CrimeDataset(train_video_paths)
    val_dataset = CrimeDataset(val_video_paths)
    test_dataset = CrimeDataset(test_video_paths)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()//2 or 1, pin_memory=True) # Use config.BATCH_SIZE
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()//2 or 1, pin_memory=True) # Use config.BATCH_SIZE
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()//2 or 1, pin_memory=True) # Use config.BATCH_SIZE
    
    main_logger.info(f"Train dataset size: {len(train_dataset)}")
    main_logger.info(f"Validation dataset size: {len(val_dataset)}")
    main_logger.info(f"Test dataset size: {len(test_dataset)}")

    model = MultimodalCrimeDetector().to(config.DEVICE) # Use config.DEVICE
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY) # Use config.LEARNING_RATE, config.WEIGHT_DECAY

    model_save_dir = config.OUTPUT_DIRS['models'] # Use config.OUTPUT_DIRS
    os.makedirs(model_save_dir, exist_ok=True)
    main_logger.info(f"Models will be saved to: {model_save_dir}")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    
    loaded_model, loaded_optimizer, loaded_epoch, loaded_loss = load_checkpoint(model, optimizer, model_save_dir, filename="best_model.pth")
    if loaded_model is not None:
        model = loaded_model
        optimizer = loaded_optimizer
        start_epoch = loaded_epoch + 1
        best_val_loss = loaded_loss
        main_logger.info(f"Resuming training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
    else:
        start_epoch = 0
        main_logger.info("No checkpoint found. Starting training from scratch.")

    main_logger.info("Starting model training...")
    for epoch in range(start_epoch, config.NUM_EPOCHS): # Use config.NUM_EPOCHS
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE, epoch, config.LOG_INTERVAL) # Use config.DEVICE, config.LOG_INTERVAL

        if (epoch + 1) % config.EVAL_INTERVAL == 0: # Use config.EVAL_INTERVAL
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_model(
                model, val_loader, criterion, config.DEVICE, "Validation" # Use config.DEVICE
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, best_val_loss, model_save_dir, filename="best_model.pth")
            else:
                epochs_no_improve += 1
                main_logger.info(f"Early stopping patience: {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE}") # Use config.EARLY_STOPPING_PATIENCE

            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE: # Use config.EARLY_STOPPING_PATIENCE
                main_logger.info(f"Early stopping triggered after {epoch+1} epochs. No improvement in validation loss for {config.EARLY_STOPPING_PATIENCE} epochs.") # Use config.EARLY_STOPPING_PATIENCE
                break
        
        if (epoch + 1) % config.SAVE_INTERVAL == 0: # Use config.SAVE_INTERVAL
            save_checkpoint(model, optimizer, epoch, train_loss, model_save_dir, filename=f"model_epoch_{epoch}.pth")
    
    main_logger.info("Training finished.")

    main_logger.info("Loading best model for final evaluation...")
    final_model = MultimodalCrimeDetector().to(config.DEVICE) # Use config.DEVICE
    final_optimizer = optim.AdamW(final_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY) # Use config.LEARNING_RATE, config.WEIGHT_DECAY
    
    loaded_final_model, _, _, _ = load_checkpoint(final_model, final_optimizer, model_save_dir, filename="best_model.pth")
    
    if loaded_final_model is None and len(val_dataset) == 0:
        last_epoch_model_path = os.path.join(model_save_dir, f"model_epoch_{config.NUM_EPOCHS-1}.pth") # Use config.NUM_EPOCHS
        if os.path.exists(last_epoch_model_path):
            main_logger.info(f"Attempting to load model from last epoch: {last_epoch_model_path}")
            loaded_final_model, _, _, _ = load_checkpoint(final_model, final_optimizer, model_save_dir, filename=f"model_epoch_{config.NUM_EPOCHS-1}.pth") # Use config.NUM_EPOCHS
    
    if loaded_final_model is None:
        main_logger.error("Could not load any trained model for final testing. Testing with freshly initialized model (may be untrained).")
    else:
        final_model = loaded_final_model
        final_model.to(config.DEVICE) # Use config.DEVICE

    if len(test_dataset) > 0:
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate_model(final_model, test_loader, criterion, config.DEVICE, "Final Test") # Use config.DEVICE
        main_logger.info("Final Test Set Metrics:")
        main_logger.info(f"  Loss: {test_loss:.4f}")
        main_logger.info(f"  Accuracy: {test_acc*100:.2f}%")
        main_logger.info(f"  Precision: {test_prec:.4f}")
        main_logger.info(f"  Recall: {test_rec:.4f}")
        main_logger.info(f"  F1 Score: {test_f1:.4f}")
        main_logger.info(f"  AUC Score: {test_auc:.4f}")
    else:
        main_logger.warning("Test dataset is empty. Skipping final evaluation.")

    main_logger.info("="*80)
    main_logger.info("TRAINING AND EVALUATION COMPLETE")

if __name__ == "__main__":
    main()