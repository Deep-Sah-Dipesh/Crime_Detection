# Crime_Detection/Code/config.py

import os
import torch

# Get the directory of the current script (e.g., Crime_Detection/Code/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume project root is one level up from SCRIPT_DIR, i.e., Crime_Detection/
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# --- Base Paths ---
# These paths are now relative to the PROJECT_ROOT (Crime_Detection/)
BASE_PATHS = {
    'processed_data': os.path.join(PROJECT_ROOT, 'Training', 'Processed_Data'),
    'raw_videos': os.path.join(PROJECT_ROOT, 'Training', 'Videos'),
    'raw_audios': os.path.join(PROJECT_ROOT, 'Training', 'Processed_Data', 'Audios'), # For audio extraction input
    'raw_transcripts': os.path.join(PROJECT_ROOT, 'Training', 'Processed_Data', 'Transcripts'), # For text extraction input
    'data_splits': os.path.join(PROJECT_ROOT, 'data_splits') # New directory for saving train/val/test splits
}

# Derived Feature Output Paths
FEATURE_PATHS = {
    'audio': os.path.join(BASE_PATHS['processed_data'], 'audio_features'),
    'video': os.path.join(BASE_PATHS['processed_data'], 'video_features'),
    'text': os.path.join(BASE_PATHS['processed_data'], 'text_features')
}

# Output Directories for logs and checkpoints (relative to PROJECT_ROOT)
OUTPUT_DIRS = {
    'logs': os.path.join(PROJECT_ROOT, 'logs'),
    'checkpoints': os.path.join(PROJECT_ROOT, 'checkpoints')
}

# --- Feature Dimensions (Derived from your feature extraction code) ---
TEXT_FEAT_DIM = 768 # RoBERTa-base embedding dimension
VIDEO_FEAT_DIM = 512 # Output dimension of your ResNet+LSTM for video features
AUDIO_FEAT_DIM = 1024 # YAMNet embedding dimension

# --- Model Hyperparameters ---
HIDDEN_DIM_TEXT = 256 # Common embedding dimension for text before fusion
HIDDEN_DIM_VIDEO = 256 # Common embedding dimension for video before fusion
HIDDEN_DIM_AUDIO = 256 # Common embedding dimension for audio before fusion
FUSION_HIDDEN_DIM = 512 # Dimension after concatenation and first fusion layer
NUM_CLASSES = 2 # Binary classification: Crime (1) / Non-Crime (0)
DROPOUT_RATE = 0.3 # Single dropout rate for all layers

# --- Training Hyperparameters ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-5 # L2 regularization
LOG_INTERVAL = 10 # Log training progress every N batches (for train_eval.py)
EVAL_INTERVAL = 1 # Evaluate on validation set every N epochs (for train_eval.py)
SAVE_INTERVAL = 5 # Save model checkpoint every N epochs (for train_eval.py)

# --- Data Processing Parameters ---
# Aggregation method for sequential features (video, audio)
SEQUENTIAL_AGGREGATION_METHOD = 'mean' # Options: 'mean', 'max'

# --- Label Mapping (CONFIRMED BASED ON XD VIOLENCE CONVENTION) ---
# _label_A in filename -> 1 (Abnormal/Violence)
# Absence of _label_A (or _label_N) -> 0 (Normal/Non-Violence)
LABEL_MAPPING = {
    '_label_A': 1, # Abnormal / Violence
    '_label_N': 0  # Normal / Non-Violence (implicitly if _label_A is not present)
}

# --- Data Split Ratios ---
# Percentages for dividing the dataset into training, validation, and testing sets
# These should sum up to 1.0
DATA_SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15,
}

# --- Misc ---
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Create all necessary directories ---
for _, path in BASE_PATHS.items():
    os.makedirs(path, exist_ok=True)
for _, path in FEATURE_PATHS.items():
    os.makedirs(path, exist_ok=True)
for _, path in OUTPUT_DIRS.items():
    os.makedirs(path, exist_ok=True)