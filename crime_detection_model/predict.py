import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import logging
logging.getLogger('tensorflow_hub').setLevel(logging.ERROR)

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import torchaudio
import tensorflow as tf
import tensorflow_hub as hub
import sys
import subprocess
import shutil
from PIL import Image
from tqdm import tqdm

try:
    import config
    from model import CrimeClassifier
except ImportError:
    print("Error: Make sure config.py and model.py are present.", file=sys.stderr)
    sys.exit(1)

# --- Feature Extraction Modules ---

class VideoFeatureExtractor(nn.Module):
    """Extracts features from video frames using a ResNet50-LSTM architecture."""
    def __init__(self):
        super(VideoFeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=2048, hidden_size=config.VIDEO_FEAT_DIM, batch_first=True)

    def forward(self, x):
        with torch.no_grad():
            cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        lstm_input = cnn_out.unsqueeze(1)
        _, (h_n, _) = self.lstm(lstm_input)
        return h_n.squeeze(0)

def extract_video_features(video_path):
    """Processes a video file frame by frame to extract aggregated features."""
    feature_extractor = VideoFeatureExtractor().to(config.DEVICE)
    feature_extractor.eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open video file {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_features = []
    
    with tqdm(total=total_frames, desc="  - Processing Video Frames", leave=False, ncols=100) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_img).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                feature = feature_extractor(input_tensor)
                frame_features.append(feature.cpu().numpy().squeeze())
            pbar.update(1)
    
    cap.release()
    if not frame_features: return torch.zeros(config.VIDEO_FEAT_DIM)
    
    return torch.from_numpy(np.mean(frame_features, axis=0)).float().squeeze()

def extract_audio_features(audio_path):
    """Extracts audio features using the local YAMNet model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_dir = os.path.join(script_dir, 'local_models', '9616fd04ec2360621642ef9455b84f4b668e219e')
    
    yamnet_model = hub.load(local_model_dir)
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    _, embeddings, _ = yamnet_model(waveform.squeeze().numpy())
    return torch.from_numpy(tf.reduce_mean(embeddings, axis=0).numpy()).float()

def extract_features(file_path):
    """Orchestrates all feature extraction for a given file."""
    video_vec = torch.zeros(config.VIDEO_FEAT_DIM)
    audio_vec = torch.zeros(config.AUDIO_FEAT_DIM)
    temp_dir = os.path.join(config.PROJECT_ROOT, "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        logging.info("-> Extracting new features...")
        video_vec = extract_video_features(file_path)
        
        audio_path = os.path.join(temp_dir, "temp_audio.mp3")
        command = [config.FFMPEG_PATH, '-i', file_path, '-q:a', '0', '-map', 'a', '-y', audio_path]
        
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio_vec = extract_audio_features(audio_path)
        
    except Exception as e:
        logging.warning(f"Could not extract audio for {os.path.basename(file_path)}. Prediction based on video only. Error: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
    text_vec = torch.zeros(config.TEXT_FEAT_DIM) 
    return torch.cat((video_vec, audio_vec, text_vec), dim=-1)

# --- Main Prediction Logic ---

def get_prediction(file_path):
    """
    Runs inference on a file, using cached features if available.
    Returns a list of all labels and their probabilities.
    """
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'.")

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    cached_feature_path = os.path.join(config.TESTING_FEATURES_DIR, f"{base_filename}_features.pt")

    if os.path.exists(cached_feature_path):
        # *** KEY FIX: Set weights_only=True to address the FutureWarning ***
        combined_features_no_batch = torch.load(cached_feature_path, weights_only=True)
    else:
        combined_features_no_batch = extract_features(file_path)
        os.makedirs(config.TESTING_FEATURES_DIR, exist_ok=True)
        torch.save(combined_features_no_batch, cached_feature_path)
        logging.info(f"   (Features saved to cache for future runs)")

    combined_features = combined_features_no_batch.unsqueeze(0).to(config.DEVICE)
    model = CrimeClassifier().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(combined_features)
        probabilities = torch.sigmoid(logits).squeeze()

    all_results = []
    for i, code in enumerate(config.ALL_LABEL_CODES):
        all_results.append((code, probabilities[i].item()))
        
    return all_results

def predict_on_file(file_path, threshold=0.5):
    """Handles the prediction process and prints formatted results."""
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    try:
        all_results = get_prediction(file_path)
        
        print(f"\n--- Analysis Results for: {os.path.basename(file_path)} ---")
        
        print(f"\n[+] Detections above {threshold:.0%} confidence:")
        detections = [res for res in all_results if res[1] >= threshold and config.LABEL_MAP.get(res[0]) != 'Normal']
        if not detections:
            print("  - No suspicious activity detected above the threshold.")
        else:
            for code, prob in detections:
                print(f"  - [!!] {config.LABEL_MAP[code]:<15} (Confidence: {prob:.2%})")

        print("\n[+] Top 3 Potential Activities:")
        top_3 = sorted(all_results, key=lambda item: item[1], reverse=True)[:3]
        for code, prob in top_3:
            print(f"  - {config.LABEL_MAP[code]:<15} (Confidence: {prob:.2%})")

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nUsage: python predict.py \"<path_to_media_file>\"")
        sys.exit(1)
    predict_on_file(sys.argv[1])

