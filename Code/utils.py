import os
import logging
from datetime import datetime
import torch
import numpy as np
import sys

_global_logger = None

def setup_logger(name, log_file, level=logging.INFO, console_output=True):
    """
    Sets up a global logger instance. This function should typically be called once
    at the start of the application. Subsequent calls will return the existing
    global logger without reconfiguring it unless a different name is provided
    and the global logger is still None.
    """
    global _global_logger
    
    # If the global logger is already set up, return it
    if _global_logger is not None:
        return _global_logger

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # Prevent messages from being handled by ancestor loggers

    # Clear existing handlers to prevent duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _global_logger = logger
    return logger

def get_logger(name):
    """
    Retrieves the global logger instance. If setup_logger has not been called yet
    to configure the global logger, it returns a standard Python logger for the
    given name.
    """
    global _global_logger
    if _global_logger is None:
        # If the global logger hasn't been set up by setup_logger,
        # return a standard logger for the given name.
        # It's recommended to call setup_logger first for custom configuration.
        return logging.getLogger(name)
    return _global_logger

def get_timestamp():
    """
    Returns the current timestamp formatted as YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def find_files_in_directory(base_dir, extensions):
    """
    Finds files with specified extensions within a directory and its subdirectories.
    Returns relative paths to the base directory.
    """
    found_files = []
    if not os.path.isdir(base_dir):
        get_logger(__name__).warning(f"Directory not found: {base_dir}. Returning empty list.")
        return []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in extensions): # Ensure extension check is also case-insensitive
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                found_files.append(relative_path)
    return found_files

def save_checkpoint(model, optimizer, epoch, loss, path, filename="best_model.pth"):
    """
    Saves a model and optimizer checkpoint.
    """
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
    """
    Loads a model and optimizer checkpoint from a specified path.
    Handles various error cases during loading.
    """
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath):
        try:
            # Load checkpoint to the appropriate device (GPU if available, else CPU)
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
    """
    Sets the random seed for reproducibility across different libraries.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    get_logger(__name__).info(f"Random seed set to {seed} for reproducibility.")