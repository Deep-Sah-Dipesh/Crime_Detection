# predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import torchaudio
from transformers import RobertaTokenizer, RobertaModel
import os
import sys
import subprocess
import shutil
from PIL import Image
from tqdm import tqdm

try:
    import config
    from model import CrimeClassifier
except ImportError:
    print("Error: Make sure config.py and model.py are in the correct directory.", file=sys.stderr)
    sys.exit(1)

# --- 1. FEATURE EXTRACTION LOGIC ---

class VideoFeatureExtractor(nn.Module):
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
    print("-> Extracting video features (ResNet50 -> LSTM)...")
    feature_extractor = VideoFeatureExtractor().to(config.DEVICE)
    feature_extractor.eval()

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_features = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Processing Video Frames") as pbar:
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
    if not frame_features:
        print("Warning: No frames extracted from video. Returning a zero vector.")
        return torch.zeros(config.VIDEO_FEAT_DIM)
    
    aggregated_features = np.mean(frame_features, axis=0)
    return torch.from_numpy(aggregated_features).float().squeeze()

def extract_audio_features(audio_path):
    print("-> Extracting audio features (YAMNet)...")
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        raise ImportError("TensorFlow/Hub not found. Please run: pip install tensorflow tensorflow-hub")

    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    _, embeddings, _ = yamnet_model(waveform.squeeze().numpy())
    return torch.from_numpy(tf.reduce_mean(embeddings, axis=0).numpy()).float()

def extract_text_features(text_content):
    print("-> Extracting text features (RoBERTa)...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base').to(config.DEVICE)
    model.eval()

    inputs = tokenizer(text_content, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    inputs = {key: val.to(config.DEVICE) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().float()

def predict_on_file(file_path, threshold=0.5):
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'. Please run train.py first.", file=sys.stderr)
        return

    video_vec = torch.zeros(config.VIDEO_FEAT_DIM)
    audio_vec = torch.zeros(config.AUDIO_FEAT_DIM)
    text_vec = torch.zeros(config.TEXT_FEAT_DIM)

    temp_dir = os.path.join(config.PROJECT_ROOT, "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"Processing VIDEO file: {os.path.basename(file_path)}")
            video_vec = extract_video_features(file_path)
            
            print("\n-> Extracting audio track...")
            audio_path = os.path.join(temp_dir, "temp_audio.mp3")
            
            # *** KEY FIX: Use the direct path to FFmpeg from config.py ***
            ffmpeg_executable = getattr(config, 'FFMPEG_PATH', 'ffmpeg') # Defaults to 'ffmpeg' if not set
            command = [ffmpeg_executable, '-i', file_path, '-q:a', '0', '-map', 'a', '-y', audio_path]
            
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                audio_vec = extract_audio_features(audio_path)
            except subprocess.CalledProcessError as e:
                print(f"   Error: FFmpeg failed. Please ensure FFMPEG_PATH in config.py is correct.")
                print(f"   FFmpeg stderr: {e.stderr.decode()}")
            except FileNotFoundError:
                 print(f"   Error: Could not find FFmpeg. Set the FFMPEG_PATH in config.py.")
            except Exception as e:
                print(f"   An unexpected error occurred during audio extraction: {e}")

        elif file_ext in ['.mp3', '.wav', '.flac']:
            print(f"Processing AUDIO file: {os.path.basename(file_path)}")
            audio_vec = extract_audio_features(file_path)

        elif file_ext == '.txt':
            print(f"Processing TEXT file: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text_vec = extract_text_features(f.read())
        else:
            print(f"Error: Unsupported file type: {file_ext}", file=sys.stderr)
            return

        model = CrimeClassifier().to(config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
        model.eval()
        print("\nModel loaded successfully.")

        combined_features = torch.cat((video_vec, audio_vec, text_vec), dim=-1).unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            logits = model(combined_features)
            probabilities = torch.sigmoid(logits).squeeze()

        print(f"\n--- Analysis Results for: {os.path.basename(file_path)} ---")
        detected_crimes = False
        for i, label_code in enumerate(config.ALL_LABEL_CODES):
            prob = probabilities[i].item()
            if prob >= threshold:
                label_name = config.LABEL_MAP[label_code]
                if label_name == 'Normal': continue # Optional: skip printing 'Normal'
                print(f"[!!] DETECTED: {label_name:<15} (Confidence: {prob:.2%})")
                detected_crimes = True
        
        if not detected_crimes:
            print("No suspicious activity detected above the threshold.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}", file=sys.stderr)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("\nCleaned up temporary files.")

def main():
    """Main function to parse arguments and run prediction."""
    if len(sys.argv) != 2:
        print("\nUsage: python predict.py \"<path_to_video_audio_or_text_file>\"")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        sys.exit(1)
        
    predict_on_file(input_file)


if __name__ == "__main__":
    main()
