import os
import logging
from datetime import datetime
import torch
import numpy as np

def setup_logger(name, log_file, level=logging.INFO, console_output=True):
    """
    Sets up a logger with both file and console handlers.
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        console_output (bool): Whether to also output logs to the console.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to prevent duplicates, especially important if called multiple times
    for handler in logger.handlers[:]: # Iterate over a copy to avoid modification issues
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def get_timestamp():
    """Returns a formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def find_files_in_directory(base_dir, extensions):
    """
    Finds all files with given extensions in a directory and its subdirectories.
    Returns relative paths from the base_dir.
    Args:
        base_dir (str): The root directory to start searching from.
        extensions (tuple): A tuple of file extensions to look for (e.g., ('.mp4', '.avi')).
    Returns:
        list: A list of relative paths to the found files.
    """
    found_files = []
    if not os.path.isdir(base_dir):
        logging.getLogger('utils').warning(f"Directory not found: {base_dir}. Returning empty list.")
        return []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                relative_path = os.path.relpath(os.path.join(root, file), base_dir)
                found_files.append(relative_path)
    return found_files


def save_checkpoint(model, optimizer, epoch, loss, path, filename="best_model.pth"):
    """
    Saves the model and optimizer state.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): Current epoch number.
        loss (float): Current validation loss.
        path (str): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    logging.getLogger('main_train_eval').info(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, path, filename="best_model.pth"):
    """
    Loads the model and optimizer state from a checkpoint.
    Args:
        model (torch.nn.Module): The model to load into.
        optimizer (torch.optim.Optimizer): The optimizer to load into.
        path (str): Directory where the checkpoint is saved.
        filename (str): Name of the checkpoint file.
    Returns:
        tuple: (model, optimizer, epoch, loss) or (None, None, None, None) if not found or corrupted.
    """
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath):
        try:
            checkpoint = torch.load(filepath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # Map to device
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            logging.getLogger('main_train_eval').info(f"Checkpoint loaded from {filepath} (Epoch: {epoch}, Loss: {loss:.4f})")
            return model, optimizer, epoch, loss
        except KeyError as e:
            logging.getLogger('main_train_eval').error(f"Error loading checkpoint from {filepath}: Missing key {e}. File might be corrupted or not a valid model checkpoint. Starting from scratch.")
            # It's crucial to return None here so the main script knows to reinitialize
            return None, None, None, None
        except Exception as e:
            logging.getLogger('main_train_eval').error(f"An unexpected error occurred while loading checkpoint from {filepath}: {e}. Starting from scratch.")
            return None, None, None, None
    logging.getLogger('main_train_eval').warning(f"No checkpoint found at {filepath}")
    return None, None, None, None