import torch
import os

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
VIDEO_DIR = os.path.join(PROJECT_ROOT, '..', 'Training', 'Videos')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'ground_truth_labels.csv')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, 'generated_models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, '..', 'Training', 'Processed_Data')
VIDEO_FEATURES_DIR = os.path.join(PROCESSED_DATA_PATH, 'video_features')
AUDIO_FEATURES_DIR = os.path.join(PROCESSED_DATA_PATH, 'audio_features')
TEXT_FEATURES_DIR = os.path.join(PROCESSED_DATA_PATH, 'text_features')
TESTING_FEATURES_DIR = os.path.join(PROJECT_ROOT, '..', 'Testing', 'Features_Extracted')

# --- External Dependencies ---
FFMPEG_PATH = r"C:\Users\admin\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# --- Feature Dimensions ---
VIDEO_FEAT_DIM = 512
AUDIO_FEAT_DIM = 1024
TEXT_FEAT_DIM = 768

# --- Model & Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_FEATURE_SIZE = VIDEO_FEAT_DIM + AUDIO_FEAT_DIM + TEXT_FEAT_DIM
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 5

# --- Labels ---
LABEL_MAP = {
    'A': 'Normal', 'B1': 'Fighting', 'B2': 'Shooting', 'B4': 'Riot',
    'B5': 'Abuse', 'B6': 'Car Accident', 'G': 'Explosion'
}
ALL_LABEL_CODES = list(LABEL_MAP.keys())
NUM_CLASSES = len(ALL_LABEL_CODES)

# --- Ensure output directories exist ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TESTING_FEATURES_DIR, exist_ok=True)

