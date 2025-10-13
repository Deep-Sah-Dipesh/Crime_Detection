import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import logging
from datetime import datetime
from tqdm import tqdm 
import heapq
import shutil
import numpy as np

import config
from model import CrimeClassifier
from dataset import CrimeFeatureDataset, collate_fn

def main():
    # Set a fixed seed for reproducibility across runs
    torch.manual_seed(42)
    np.random.seed(42)

    log_filename = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(config.LOG_DIR, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()]
    )

    logging.info("Loading dataset...")
    full_dataset = CrimeFeatureDataset(annotations_csv=config.OUTPUT_CSV)
    val_size = int(config.VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
    logging.info(f"Data loaded: {train_size} training, {val_size} validation samples.")

    model = CrimeClassifier().to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    logging.info(f"--- Starting Training on {config.DEVICE} for up to {config.NUM_EPOCHS} epochs ---")

    best_models = [] # Min-heap to store top 3 models (val_loss, train_loss, path)
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        for features, labels in train_pbar:
            if features.nelement() == 0: continue
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                if features.nelement() == 0: continue
                features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader) if train_loader else 0
        avg_val_loss = total_val_loss / len(val_loader) if val_loader else 0
        scheduler.step(avg_val_loss)
        
        logging.info(f"Epoch [{epoch+1:02d}/{config.NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        current_best_loss = -best_models[0][0] if best_models and len(best_models) >= 3 else float('inf')

        if avg_val_loss < current_best_loss:
            logging.info(f"  -> Val loss improved to {avg_val_loss:.4f}. Managing saved models.")
            temp_model_path = os.path.join(config.MODEL_SAVE_DIR, f"temp_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), temp_model_path)
            
            # Use a max-heap approach with negative loss to track top 3
            heapq.heappush(best_models, (-avg_val_loss, avg_train_loss, temp_model_path))
            if len(best_models) > 3:
                _, _, old_model_path = heapq.heappop(best_models)
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"  -> Val loss did not improve. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logging.info("--- Early stopping triggered. ---")
            break
            
    # Finalize saved models by renaming them
    sorted_models = sorted([heapq.heappop(best_models) for _ in range(len(best_models))], reverse=True)
    for i, (neg_val_loss, train_loss, path) in enumerate(sorted_models):
        rank = i + 1
        final_name = f"best_model_{rank}.pth" if rank > 1 else "best_model.pth"
        final_path = os.path.join(config.MODEL_SAVE_DIR, final_name)
        shutil.move(path, final_path)
        logging.info(f"Saved Rank {rank} model as {final_name} (Val Loss: {-neg_val_loss:.4f}, Train Loss: {train_loss:.4f})")
    
    logging.info(f"--- Training Finished. Best models saved in {config.MODEL_SAVE_DIR} ---")

if __name__ == "__main__":
    main()

