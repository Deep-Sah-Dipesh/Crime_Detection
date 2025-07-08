import os
import logging
from datetime import datetime
import torch
import numpy as np
import sys

_global_logger = None

def setup_logger(name, log_file, level=logging.INFO, console_output=True):
    global _global_logger
    if _global_logger is not None:
        return _global_logger

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _global_logger = logger
    return logger

def get_logger(name):
    global _global_logger
    if _global_logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)
    return _global_logger

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def find_files_in_directory(base_dir, extensions):
    found_files = []
    if not os.path.isdir(base_dir):
        get_logger(__name__).warning(f"Directory not found: {base_dir}. Returning empty list.")
        return []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                found_files.append(relative_path)
    return found_files

def save_checkpoint(model, optimizer, epoch, loss, path, filename="best_model.pth"):
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    get_logger(__name__).info(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, path, filename="best_model.pth"):
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath):
        try:
            checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            get_logger(__name__).info(f"Checkpoint loaded from {filepath} (Epoch: {epoch}, Loss: {loss:.4f})")
            return model, optimizer, epoch, loss
        except KeyError as e:
            get_logger(__name__).error(f"Error loading checkpoint from {filepath}: Missing key {e}. File might be corrupted or not a valid model checkpoint. Starting from scratch.")
            return None, None, None, None
        except Exception as e:
            get_logger(__name__).error(f"An unexpected error occurred while loading checkpoint from {filepath}: {e}. Starting from scratch.")
            return None, None, None, None
    get_logger(__name__).warning(f"No checkpoint found at {filepath}")
    return None, None, None, None

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    get_logger(__name__).info(f"Random seed set to {seed} for reproducibility.")