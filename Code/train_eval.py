# Crime_Detection/Code/train_eval.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split # For stratified splitting
import numpy as np
import time
import logging
from tqdm import tqdm
import json # To save/load data splits

# Adjust sys.path to import from project root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all necessary constants from config.py
from config import BASE_PATHS, OUTPUT_DIRS, \
                   TEXT_FEAT_DIM, VIDEO_FEAT_DIM, AUDIO_FEAT_DIM, \
                   HIDDEN_DIM_TEXT, HIDDEN_DIM_VIDEO, HIDDEN_DIM_AUDIO, \
                   FUSION_HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE, \
                   BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, \
                   EARLY_STOPPING_PATIENCE, WEIGHT_DECAY, LOG_INTERVAL, \
                   EVAL_INTERVAL, SAVE_INTERVAL, \
                   SEQUENTIAL_AGGREGATION_METHOD, LABEL_MAPPING, \
                   RANDOM_SEED, DEVICE

from Code.dataset import CrimeDataset
from Code.model import MultimodalCrimeDetector
from Code.utils import setup_logger, save_checkpoint, load_checkpoint, get_timestamp, find_files_in_directory

import torch.nn.functional as F

# --- Setup Logging ---
log_dir = OUTPUT_DIRS['logs']
os.makedirs(log_dir, exist_ok=True)
timestamp = get_timestamp()
main_logger = setup_logger('main_train_eval', os.path.join(log_dir, f'train_eval_{timestamp}.log'))
main_logger.info("Starting training/evaluation script.")
main_logger.info(f"Using device: {DEVICE}")

# --- Set Random Seed for Reproducibility ---
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
main_logger.info(f"Random seed set to {RANDOM_SEED} for reproducibility.")


def get_label_from_filename(filename, label_mapping):
    """
    Extracts the numerical label from the filename based on the label mapping.
    Assumes filename contains '_label_A' or '_label_N'.
    Returns the corresponding numerical label.
    """
    if '_label_A' in filename:
        return label_mapping['_label_A']
    return label_mapping['_label_N'] 


def get_data_splits(base_video_dir, data_splits_dir, split_ratios, label_mapping, force_recreate=False):
    """
    Loads or creates stratified train/val/test splits based on video filenames.
    Saves the splits to JSON files for reproducibility.
    """
    os.makedirs(data_splits_dir, exist_ok=True) # Ensure the data_splits directory exists
    splits_file = os.path.join(data_splits_dir, 'data_splits.json')
    
    if os.path.exists(splits_file) and not force_recreate:
        main_logger.info(f"Loading existing data splits from {splits_file}")
        try:
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            if all(k in splits and isinstance(splits[k], list) for k in ['train', 'val', 'test']):
                main_logger.info(f"Loaded splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
                return splits['train'], splits['val'], splits['test']
            else:
                main_logger.warning("Loaded data splits file is malformed. Recreating splits.")
        except json.JSONDecodeError as e:
            main_logger.warning(f"Error decoding data splits JSON: {e}. Recreating splits.")
        except Exception as e:
            main_logger.warning(f"Unexpected error loading data splits: {e}. Recreating splits.")
    
    main_logger.info("Creating new data splits...")
    all_video_clip_paths = find_files_in_directory(base_video_dir, ('.mp4', '.avi', '.mov', '.mkv'))
    
    if not all_video_clip_paths:
        main_logger.error(f"No video files found in {base_video_dir} for splitting.")
        return [], [], []

    # Extract labels for stratification
    clip_labels = []
    valid_clips_with_labels = []
    for clip_path in all_video_clip_paths:
        label = get_label_from_filename(os.path.basename(clip_path), label_mapping)
        clip_labels.append(label)
        valid_clips_with_labels.append(clip_path)

    if not valid_clips_with_labels:
        main_logger.error("No clips with recognized labels found for splitting.")
        return [], [], []

    main_logger.info(f"Found {len(valid_clips_with_labels)} clips with labels for splitting.")
    unique_labels, label_counts = np.unique(clip_labels, return_counts=True)
    main_logger.info(f"Label distribution for splitting: {dict(zip(unique_labels, label_counts))}")

    # Stratified split for train/test
    test_size_actual = split_ratios['test']
    # Adjust test_size if it's too small for stratification (requires at least 2 samples per class)
    if len(np.unique(clip_labels)) > 1 and min(label_counts) < 2:
        main_logger.warning("One or more classes have fewer than 2 samples. Stratification might fail or be less effective.")
    
    # Ensure test_size is not so large that it leaves too few samples for train/val
    if len(valid_clips_with_labels) * (split_ratios['train'] + split_ratios['val']) < 2:
        main_logger.warning("Not enough total samples for train/val. Adjusting test_size to ensure at least 2 samples for train/val.")
        test_size_actual = max(0.01, (len(valid_clips_with_labels) - 2) / len(valid_clips_with_labels))
        main_logger.warning(f"Adjusted test_size to {test_size_actual:.2f}")

    train_val_clips, test_clips, train_val_labels, _ = train_test_split(
        valid_clips_with_labels, clip_labels, 
        test_size=test_size_actual, 
        stratify=clip_labels, 
        random_state=RANDOM_SEED
    )

    # Stratified split for train/val from train_val_set
    relative_val_size = split_ratios['val'] / (split_ratios['train'] + split_ratios['val'])
    if len(train_val_clips) * split_ratios['train'] < 2:
         main_logger.warning("Not enough samples for training set. Adjusting val_size to ensure at least 2 samples for train.")
         relative_val_size = max(0.01, (len(train_val_clips) - 2) / len(train_val_clips))
         main_logger.warning(f"Adjusted relative_val_size to {relative_val_size:.2f}")

    train_clips, val_clips, _, _ = train_test_split(
        train_val_clips, train_val_labels, 
        test_size=relative_val_size, 
        stratify=train_val_labels, 
        random_state=RANDOM_SEED
    )

    # Save splits
    splits = {
        'train': train_clips,
        'val': val_clips,
        'test': test_clips
    }
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=4)
    main_logger.info(f"Data splits saved to {splits_file}")
    
    main_logger.info(f"Train clips: {len(train_clips)}, Val clips: {len(val_clips)}, Test clips: {len(test_clips)}")
    return train_clips, val_clips, test_clips


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} Training")):
        text_feats = batch_data['text_features'].to(device)
        audio_feats = batch_data['audio_features'].to(device)
        video_feats = batch_data['video_features'].to(device)
        labels = batch_data['label'].to(device)

        optimizer.zero_grad()
        
        outputs = model(text_feats, video_feats, audio_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            main_logger.debug(f"Epoch [{epoch}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], "
                              f"Loss: {loss.item():.4f}, Batch Acc: {(correct_predictions/(total_samples+1e-6))*100:.2f}%")
            
    avg_loss = total_loss / len(dataloader)
    epoch_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0
    main_logger.info(f"Epoch {epoch} Training Summary: Avg Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    return avg_loss, epoch_accuracy

def evaluate_model(model, dataloader, criterion, device, epoch_or_type="Validation"):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probabilities = [] # For AUC-ROC

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"{epoch_or_type} Evaluation"):
            text_feats = batch_data['text_features'].to(device)
            audio_feats = batch_data['audio_features'].to(device)
            video_feats = batch_data['video_features'].to(device)
            labels = batch_data['label'].to(device)

            outputs = model(text_feats, video_feats, audio_feats)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            if NUM_CLASSES > 1: 
                all_probabilities.extend(probabilities[:, LABEL_MAPPING['_label_A']].cpu().numpy()) 
            else:
                all_probabilities.extend(probabilities.cpu().numpy())


    avg_loss = total_loss / len(dataloader)
    
    if not all_labels:
        main_logger.warning(f"No samples processed for {epoch_or_type} evaluation. Metrics set to 0.0.")
        return avg_loss, 0.0, 0.0, 0.0, 0.0, 0.0

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0, average='binary')
    recall = recall_score(all_labels, all_predictions, zero_division=0, average='binary')
    f1 = f1_score(all_labels, all_predictions, zero_division=0, average='binary')
    
    roc_auc = 0.0
    try:
        if len(np.unique(all_labels)) > 1:
            roc_auc = roc_auc_score(all_labels, all_probabilities)
        else:
            main_logger.warning(f"Only one class present in {epoch_or_type} labels, AUC-ROC is not defined. Setting to 0.0.")
    except ValueError as e:
        main_logger.warning(f"Error calculating AUC-ROC for {epoch_or_type}: {e}. Setting to 0.0.")
    except Exception as e:
        main_logger.warning(f"Unexpected error calculating AUC-ROC for {epoch_or_type}: {e}. Setting to 0.0.")


    main_logger.info(f"{epoch_or_type} Results: Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%, "
                     f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC-ROC: {roc_auc:.4f}")
    return avg_loss, accuracy, precision, recall, f1, roc_auc


def main():
    # --- Data Loading and Splitting ---
    train_clip_paths, val_clip_paths, test_clip_paths = get_data_splits(
        base_video_dir=BASE_PATHS['raw_videos'], 
        data_splits_dir=BASE_PATHS['data_splits'], 
        split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}, # Using fixed ratios for splitting function
        label_mapping=LABEL_MAPPING,
        force_recreate=False # Set to True to force recreation of splits
    )

    # Initialize datasets
    main_logger.info("Initializing datasets...")
    train_dataset = CrimeDataset(video_sub_paths=train_clip_paths)
    val_dataset = CrimeDataset(video_sub_paths=val_clip_paths)
    test_dataset = CrimeDataset(video_sub_paths=test_clip_paths)

    # Check if datasets are empty after filtering missing features
    if len(train_dataset) == 0:
        main_logger.critical("Training dataset is EMPTY after filtering missing features. Cannot proceed with training. Ensure features are extracted.")
        return
    
    if len(val_dataset) == 0:
        main_logger.warning("Validation dataset is EMPTY after filtering missing features. Validation will be skipped.")
    
    if len(test_dataset) == 0:
        main_logger.warning("Test dataset is EMPTY after filtering missing features. Test evaluation will be skipped.")

    # Initialize DataLoaders
    num_dataloader_workers = min(os.cpu_count(), 8)
    main_logger.info(f"Using {num_dataloader_workers} workers for DataLoaders.")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_dataloader_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_dataloader_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_dataloader_workers, pin_memory=True)
    
    main_logger.info(f"Actual Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    main_logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # --- Model, Optimizer, Loss ---
    model = MultimodalCrimeDetector().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    main_logger.info(f"Model architecture:\n{model}")
    main_logger.info(f"Optimizer: {optimizer.__class__.__name__}, LR: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    main_logger.info(f"Loss function: {criterion.__class__.__name__}")

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    model_save_dir = OUTPUT_DIRS['checkpoints']
    os.makedirs(model_save_dir, exist_ok=True)
    
    loaded_model, loaded_optimizer, loaded_epoch, loaded_loss = load_checkpoint(
        model, optimizer, model_save_dir, filename="best_model.pth"
    )
    if loaded_epoch is not None:
        model = loaded_model 
        optimizer = loaded_optimizer
        start_epoch = loaded_epoch + 1
        best_val_loss = loaded_loss
        main_logger.info(f"Resuming training from epoch {start_epoch} with previous best validation loss {best_val_loss:.4f}")
    
    model.to(DEVICE) # Ensure the model is on the correct device

    # --- Training Loop ---
    main_logger.info("Starting training loop...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = float('inf'), 0.0, 0.0, 0.0, 0.0, 0.0
        if len(val_dataset) > 0 and (epoch + 1) % EVAL_INTERVAL == 0:
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_model(model, val_loader, criterion, DEVICE, f"Epoch {epoch} Validation")

        epoch_end_time = time.time()
        main_logger.info(f"Epoch {epoch} finished in {(epoch_end_time - epoch_start_time):.2f} seconds.")

        # Early Stopping and Checkpoint Saving
        if len(val_dataset) > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(model, optimizer, epoch, best_val_loss, model_save_dir, filename="best_model.pth")
                main_logger.info(f"New best model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                main_logger.info(f"Early stopping patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    main_logger.info(f"Early stopping triggered at epoch {epoch}. No improvement in validation loss for {EARLY_STOPPING_PATIENCE} epochs.")
                    break
        else:
            if (epoch + 1) % SAVE_INTERVAL == 0:
                save_checkpoint(model, optimizer, epoch, train_loss, model_save_dir, filename=f"model_epoch_{epoch}.pth")
                main_logger.info(f"Model checkpoint saved at epoch {epoch} (no validation set). Train loss: {train_loss:.4f}")
            main_logger.warning("No validation set available. Early stopping based on validation loss is disabled. Saving periodically.")

    main_logger.info("Training complete.")

    # --- Final Evaluation on Test Set ---
    main_logger.info("Loading best model for final test set evaluation...")
    final_model = MultimodalCrimeDetector().to(DEVICE) # Initialize a fresh model
    
    loaded_final_model, _, _, _ = load_checkpoint(final_model, optimizer, model_save_dir, filename="best_model.pth")
    
    if loaded_final_model is None and len(val_dataset) == 0: # If no best_model.pth and no val set, try last epoch's model
        last_epoch_model_path = os.path.join(model_save_dir, f"model_epoch_{NUM_EPOCHS-1}.pth")
        if os.path.exists(last_epoch_model_path):
            main_logger.info(f"Attempting to load model from last epoch: {last_epoch_model_path}")
            loaded_final_model, _, _, _ = load_checkpoint(final_model, optimizer, model_save_dir, filename=f"model_epoch_{NUM_EPOCHS-1}.pth")
    
    if loaded_final_model is None:
        main_logger.error("Could not load any trained model for final testing. Testing with freshly initialized model (may be untrained).")
    else:
        final_model = loaded_final_model
        final_model.to(DEVICE)

    if len(test_dataset) > 0:
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate_model(final_model, test_loader, criterion, DEVICE, "Final Test")
        main_logger.info("Final Test Set Metrics:")
        main_logger.info(f"  Loss: {test_loss:.4f}")
        main_logger.info(f"  Accuracy: {test_acc*100:.2f}%")
        main_logger.info(f"  Precision: {test_prec:.4f}")
        main_logger.info(f"  Recall: {test_rec:.4f}")
        main_logger.info(f"  F1-Score: {test_f1:.4f}")
        main_logger.info(f"  AUC-ROC: {test_auc:.4f}")
    else:
        main_logger.warning("No test set available for final evaluation.")

    main_logger.info("Script finished.")

if __name__ == '__main__':
    main()