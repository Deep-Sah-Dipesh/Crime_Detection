import os
import torch

class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

    BASE_PATHS = {
        'processed_data': os.path.join(PROJECT_ROOT, 'Training', 'Processed_Data'),
        'data_splits': os.path.join(PROJECT_ROOT, 'data_splits')
    }

    FEATURE_PATHS = {
        'audio': os.path.join(BASE_PATHS['processed_data'], 'audio_features'),
        'video': os.path.join(BASE_PATHS['processed_data'], 'video_features'),
        'text': os.path.join(BASE_PATHS['processed_data'], 'text_features')
    }

    OUTPUT_DIRS = {
        'logs': os.path.join(PROJECT_ROOT, 'logs'),
        'checkpoints': os.path.join(PROJECT_ROOT, 'checkpoints')
    }

    TEXT_FEAT_DIM = 768
    VIDEO_FEAT_DIM = 512
    AUDIO_FEAT_DIM = 1024

    HIDDEN_DIM_TEXT = 256
    HIDDEN_DIM_VIDEO = 256
    HIDDEN_DIM_AUDIO = 256
    FUSION_HIDDEN_DIM = 512
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.3

    BATCH_SIZE = 16
    SEQUENTIAL_AGGREGATION_METHOD = 'mean'
    
    LABEL_MAPPING = {
        '_label_A': 1,
        '_label_N': 0
    }

    MODEL_SAVE_PATH = OUTPUT_DIRS['checkpoints']
    CHECKPOINT_FILENAME = "best_model.pth"
    
    INFERENCE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'Inference_Results')

    FEATURE_METADATA_FILE = os.path.join(BASE_PATHS['processed_data'], 'feature_metadata.json')

    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = min(os.cpu_count() or 1, 8)

os.makedirs(Config.BASE_PATHS['processed_data'], exist_ok=True)
os.makedirs(Config.BASE_PATHS['data_splits'], exist_ok=True)
os.makedirs(Config.OUTPUT_DIRS['logs'], exist_ok=True)
os.makedirs(Config.OUTPUT_DIRS['checkpoints'], exist_ok=True)
os.makedirs(Config.INFERENCE_OUTPUT_PATH, exist_ok=True)