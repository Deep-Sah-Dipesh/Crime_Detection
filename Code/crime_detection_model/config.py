# config.py
import torch
import os

# --- Dynamic Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'Training', 'Processed_Data')
FFMPEG_PATH = r"C:\Users\admin\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# --- Core Paths ---
VIDEO_DIR = os.path.join(PROJECT_ROOT, 'Training', 'Videos')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'ground_truth_labels.csv')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'detection_models')

# --- Feature Paths ---
VIDEO_FEATURES_DIR = os.path.join(PROCESSED_DATA_PATH, 'video_features')
AUDIO_FEATURES_DIR = os.path.join(PROCESSED_DATA_PATH, 'audio_features')
TEXT_FEATURES_DIR = os.path.join(PROCESSED_DATA_PATH, 'text_features')

# --- CORRECTED Feature Dimensions ---
# These now match your pre-extracted features
VIDEO_FEAT_DIM = 512
AUDIO_FEAT_DIM = 1024
TEXT_FEAT_DIM = 768

# --- Model & Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# This will now correctly calculate to 2304
INPUT_FEATURE_SIZE = VIDEO_FEAT_DIM + AUDIO_FEAT_DIM + TEXT_FEAT_DIM
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 10

# --- Labels ---
LABEL_MAP = {
    'A': 'Normal', 'B1': 'Fighting', 'B2': 'Shooting', 'B4': 'Riot',
    'B5': 'Abuse', 'B6': 'Car Accident', 'G': 'Explosion'
}
ALL_LABEL_CODES = list(LABEL_MAP.keys())
NUM_CLASSES = len(ALL_LABEL_CODES)

# --- Ensure output directories exist ---
os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)