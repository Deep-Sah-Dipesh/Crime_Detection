import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'Training', 'Processed_Data')

BASE_PATHS = {
    'processed_data': PROCESSED_DATA_PATH,
    'raw_videos': os.path.join(PROJECT_ROOT, 'Training', 'Videos'),
    'raw_audios': os.path.join(PROCESSED_DATA_PATH, 'Audios'),
    'raw_transcripts': os.path.join(PROCESSED_DATA_PATH, 'Transcripts'),
    'data_splits': os.path.join(PROJECT_ROOT, 'data_splits')
}

FEATURE_PATHS = {
    'audio': os.path.join(BASE_PATHS['processed_data'], 'audio_features'),
    'video': os.path.join(BASE_PATHS['processed_data'], 'video_features'),
    'text': os.path.join(BASE_PATHS['processed_data'], 'text_features')
}

OUTPUT_DIRS = {
    'logs': os.path.join(PROJECT_ROOT, 'logs'),
    'models': os.path.join(PROJECT_ROOT, 'models'),
    'plots': os.path.join(PROJECT_ROOT, 'plots')
}

for path in OUTPUT_DIRS.values():
    os.makedirs(path, exist_ok=True)

AUDIO_FEAT_DIM = 1024
TEXT_FEAT_DIM = 768
VIDEO_FEAT_DIM = 512

HIDDEN_DIM_TEXT = 256
HIDDEN_DIM_VIDEO = 256
HIDDEN_DIM_AUDIO = 256
FUSION_HIDDEN_DIM = 512
NUM_CLASSES = 2
DROPOUT_RATE = 0.3

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-5
LOG_INTERVAL = 10
EVAL_INTERVAL = 1
SAVE_INTERVAL = 5

SEQUENTIAL_AGGREGATION_METHOD = 'mean'

LABEL_MAPPING = {
    '_label_A': 1,
    '_label_N': 0
}

DATA_SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

RANDOM_SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")