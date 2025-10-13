import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure TensorFlow logging after the initial setup
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import torchaudio
import tensorflow as tf
import tensorflow_hub as hub
import subprocess
import shutil
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime

try:
    import config
    from model import CrimeClassifier
except ImportError as e:
    print(f"Error: Failed to import config.py or model.py.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)

# --- Logging Setup ---
def setup_logging(log_dir):
    """Initializes a logger to save run information to a file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"compare_log_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger('crime_compare')
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    # Clear existing handlers to avoid duplicate messages
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger, log_filepath

# --- DUPLICATED FEATURE EXTRACTION LOGIC (Standalone) ---

class VideoFeatureExtractor(nn.Module):
    """Extracts features from video frames using a ResNet50-LSTM architecture."""
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=2048, hidden_size=config.VIDEO_FEAT_DIM, batch_first=True)

    def forward(self, x):
        with torch.no_grad(): cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        lstm_input = cnn_out.unsqueeze(1)
        _, (h_n, _) = self.lstm(lstm_input)
        return h_n.squeeze(0)

def _extract_video_features(video_path):
    """Processes a video file frame by frame to extract aggregated features."""
    feature_extractor = VideoFeatureExtractor().to(config.DEVICE)
    feature_extractor.eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video file {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total_frames), desc="  - Processing Video Frames", ncols=100, leave=False):
        ret, frame = cap.read()
        if not ret: break
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            feature = feature_extractor(input_tensor)
            frames.append(feature.cpu().numpy().squeeze())
    cap.release()
    if not frames: return torch.zeros(config.VIDEO_FEAT_DIM)
    return torch.from_numpy(np.mean(frames, axis=0)).float()

def _extract_audio_features(audio_path):
    """Extracts audio features using a local YAMNet model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_local_models_dir = os.path.join(script_dir, "local_models")
    try:
        subdirectories = [d for d in os.listdir(base_local_models_dir) if os.path.isdir(os.path.join(base_local_models_dir, d))]
        local_model_path = os.path.join(base_local_models_dir, subdirectories[0])
    except (FileNotFoundError, IndexError):
         raise FileNotFoundError(f"YAMNet model directory not found in '{base_local_models_dir}'.")
    yamnet_model = hub.load(local_model_path)
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
    _, embeddings, _ = yamnet_model(waveform.squeeze().numpy())
    return torch.from_numpy(tf.reduce_mean(embeddings, axis=0).numpy()).float()

def _extract_all_features(file_path):
    """Orchestrates all feature extraction for the comparison."""
    temp_dir = os.path.join(config.PROJECT_ROOT, "temp_processing_compare")
    os.makedirs(temp_dir, exist_ok=True)
    video_vec = torch.zeros(config.VIDEO_FEAT_DIM)
    audio_vec = torch.zeros(config.AUDIO_FEAT_DIM)
    text_vec = torch.zeros(config.TEXT_FEAT_DIM) # Text remains a placeholder
    try:
        video_vec = _extract_video_features(file_path)
        audio_path = os.path.join(temp_dir, "temp_audio.mp3")
        ffmpeg_executable = getattr(config, 'FFMPEG_PATH', 'ffmpeg')
        command = [ffmpeg_executable, '-i', file_path, '-q:a', '0', '-map', 'a', '-y', audio_path]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio_vec = _extract_audio_features(audio_path)
    except Exception as e:
        logger.warning(f"Could not extract audio. Analysis based on video only. Error: {e}")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    return video_vec, audio_vec, text_vec

def _run_inference(video_vec, audio_vec, text_vec):
    """Loads the model and runs inference on a given set of feature vectors."""
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'.")
    combined_features = torch.cat((video_vec, audio_vec, text_vec), dim=-1).unsqueeze(0).to(config.DEVICE)
    model = CrimeClassifier().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(combined_features)
        probabilities = torch.sigmoid(logits).squeeze()
    return [(code, probabilities[i].item()) for i, code in enumerate(config.ALL_LABEL_CODES)]

# --- MAIN ANALYSIS SCRIPT ---

def format_predictions(results):
    """Formats a list of prediction results into a readable string."""
    top_3 = sorted(results, key=lambda item: item[1], reverse=True)[:3]
    return ", ".join([f"{config.LABEL_MAP.get(code, 'N/A')}: {prob:.1%}" for code, prob in top_3])

def analyze_file_in_detail(file_path, logger):
    """
    Runs predictions using individual and combined features and logs a summary.
    """
    logger.info(f"--- Starting Detailed Analysis for: {os.path.basename(file_path)} ---\n")

    logger.info("-> Extracting features from file (this may take a moment)...")
    video_vec, audio_vec, text_vec = _extract_all_features(file_path)

    logger.info("-> Running predictions on each data source...\n")

    zeros_video = torch.zeros(config.VIDEO_FEAT_DIM)
    zeros_audio = torch.zeros(config.AUDIO_FEAT_DIM)
    zeros_text = torch.zeros(config.TEXT_FEAT_DIM)

    video_only_results = _run_inference(video_vec, zeros_audio, zeros_text)
    audio_only_results = _run_inference(zeros_video, audio_vec, zeros_text)
    text_only_results = _run_inference(zeros_video, zeros_audio, text_vec)
    combined_results = _run_inference(video_vec, audio_vec, text_vec)

    summary_data = {
        "Data Source": ["Video Only", "Audio Only", "Text Only", "Combined (Multimodal)"],
        "Top Predictions": [
            format_predictions(video_only_results),
            format_predictions(audio_only_results),
            format_predictions(text_only_results),
            format_predictions(combined_results)
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    header = "="*70
    title = " " * 17 + "Unimodal vs. Multimodal Predictions"
    report_string = f"{header}\n{title}\n{header}\n{summary_df.to_string(index=False)}\n{header}\n"
    logger.info(report_string)

    logger.info("--- Analysis Summary ---\n")
    video_top_pred = config.LABEL_MAP.get(sorted(video_only_results, key=lambda r: r[1], reverse=True)[0][0])
    audio_top_pred = config.LABEL_MAP.get(sorted(audio_only_results, key=lambda r: r[1], reverse=True)[0][0])
    combined_top_pred = config.LABEL_MAP.get(sorted(combined_results, key=lambda r: r[1], reverse=True)[0][0])

    if video_top_pred == combined_top_pred and audio_top_pred == combined_top_pred:
        logger.info(f"All modalities strongly agree. The model is highly confident that the activity is '{combined_top_pred}'.")
    elif video_top_pred == combined_top_pred or audio_top_pred == combined_top_pred:
        logger.info("One modality (either video or audio) was dominant in the final prediction.")
        logger.info(f"The video suggested '{video_top_pred}', the audio suggested '{audio_top_pred}', and the combined result was '{combined_top_pred}'.")
    else:
        logger.info("The combination of video and audio features led to a different conclusion than either modality alone.")
        logger.info(f"The video suggested '{video_top_pred}', the audio suggested '{audio_top_pred}', but the combined evidence pointed to '{combined_top_pred}'.")
        logger.info("This highlights the power of multimodal analysis.")
    logger.info("\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nUsage: python compare.py \"<path_to_media_file>\"")
        sys.exit(1)

    file_to_analyze = sys.argv[1]
    if not os.path.exists(file_to_analyze):
        print(f"Error: File not found at '{file_to_analyze}'", file=sys.stderr)
        sys.exit(1)

    # Setup logger for this run
    logger, log_file = setup_logging(config.LOG_DIR)
    logger.info(f"Log file for this run: {log_file}")
    
    try:
        analyze_file_in_detail(file_to_analyze, logger)
    except Exception as e:
        logger.error("An unexpected error occurred during analysis.", exc_info=True)

