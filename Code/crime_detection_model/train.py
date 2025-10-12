# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import logging
from datetime import datetime
from tqdm import tqdm 

# --- Local Imports ---
import config
from model import CrimeClassifier
from dataset import CrimeFeatureDataset, collate_fn

# --- Main training function ---
def main():
    # 1. SETUP LOGGING
    log_dir = os.path.join(config.PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"crime_det_model_train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )

    # 2. DATA LOADING
    logging.info("Loading dataset from CSV...")
    full_dataset = CrimeFeatureDataset(annotations_csv=config.OUTPUT_CSV)

    val_size = int(config.VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # KEY FIX: Set num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
    logging.info(f"Data loaded: {train_size} training samples, {592} validation samples.")

    # 3. MODEL INITIALIZATION
    model = CrimeClassifier().to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    logging.info(f"--- Starting Training on {config.DEVICE} for {config.NUM_EPOCHS} epochs ---")
    logging.info(f"Model architecture: \n{model}")

    # 4. TRAINING LOOP
    best_val_loss = float('inf')
    patience_counter = 0

    try:
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            total_train_loss = 0
            # ADDED: tqdm progress bar for the training loop
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
                # Update progress bar description with current loss
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            # --- Validation Phase ---
            model.eval()
            total_val_loss = 0
            # ADDED: tqdm progress bar for the validation loop
            val_pbar = tqdm(val_loader, desc="Validating")
            with torch.no_grad():
                for features, labels in val_pbar:
                    if features.nelement() == 0: continue
                    features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            logging.info(f"Epoch [{epoch+1:02d}/{config.NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
                torch.save(model.state_dict(), model_path)
                logging.info(f"  -> Val loss improved to {avg_val_loss:.4f}. Model saved.")
                patience_counter = 0
            else:
                patience_counter += 1
                logging.info(f"  -> Val loss did not improve. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logging.info("--- Early stopping triggered. ---")
                break
    except Exception:
        logging.error("An error occurred during training.", exc_info=True)
    finally:
        logging.info(f"\n--- Training Finished. Best model is saved at {os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth')} ---")

if __name__ == "__main__":
    main()