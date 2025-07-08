import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import logging

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import VIDEO_FEAT_DIM
from utils import get_logger

video_extractor_logger = get_logger('VideoExtractor')

class VideoFeatureExtractor:
    def __init__(self, frame_rate=1, sequence_length=16, device=None):
        self.frame_rate = frame_rate
        self.sequence_length = sequence_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cnn_model = self._load_cnn_model()
        self.lstm_model = self._load_lstm_model()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        video_extractor_logger.info(f"VideoFeatureExtractor initialized on {self.device}.")

    def _load_cnn_model(self):
        cnn_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])
        for param in cnn_model.parameters():
            param.requires_grad = False
        cnn_model.eval()
        return cnn_model.to(self.device)

    def _load_lstm_model(self):
        class SimpleLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=VIDEO_FEAT_DIM):
                super(SimpleLSTM, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])

        lstm_input_dim = 2048
        lstm_hidden_dim = 512
        lstm_model = SimpleLSTM(lstm_input_dim, lstm_hidden_dim, output_dim=VIDEO_FEAT_DIM)
        return lstm_model.to(self.device)

    def extract(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            video_extractor_logger.error(f"Failed to open video file: {video_path}")
            return None

        frame_features = []
        current_sequence = []
        frame_count = 0

        try:
            with torch.no_grad():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % self.frame_rate == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                        
                        cnn_feature = self.cnn_model(input_tensor).squeeze().detach().cpu().numpy()
                        
                        current_sequence.append(cnn_feature)
                        
                        if len(current_sequence) == self.sequence_length:
                            sequence_tensor = torch.tensor(np.array(current_sequence), dtype=torch.float32).unsqueeze(0).to(self.device)
                            lstm_output = self.lstm_model(sequence_tensor).squeeze(0).detach().cpu().numpy()
                            frame_features.append(lstm_output)
                            current_sequence = []
                            
                    frame_count += 1
            
            if len(current_sequence) > 0:
                while len(current_sequence) < self.sequence_length:
                    current_sequence.append(np.zeros(2048, dtype=np.float32))
                
                sequence_tensor = torch.tensor(np.array(current_sequence), dtype=torch.float32).unsqueeze(0).to(self.device)
                lstm_output = self.lstm_model(sequence_tensor).squeeze(0).detach().cpu().numpy()
                frame_features.append(lstm_output)

            if not frame_features:
                video_extractor_logger.warning(f"No features extracted for {video_path}. Video might be too short or corrupted.")
                return None

            all_features = np.array(frame_features)
            return all_features

        except Exception as e:
            video_extractor_logger.error(f"Error processing video {video_path}: {e}")
            return None
        finally:
            cap.release()

def extract_video_features(video_path, output_dir, filename_base, extractor_instance):
    if not os.path.exists(video_path):
        video_extractor_logger.warning(f"Video file not found: {video_path}. Skipping.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"{filename_base}_all.npy")

    try:
        features = extractor_instance.extract(video_path)
        if features is None:
            video_extractor_logger.error(f"Failed to extract features from {video_path} (returned None).")
            np.save(output_filepath, np.zeros((1, VIDEO_FEAT_DIM), dtype=np.float32))
            return False

        np.save(output_filepath, features)
        return True
        
    except Exception as e:
        video_extractor_logger.error(f"Error during video feature extraction for {video_path}: {e}")
        np.save(output_filepath, np.zeros((1, VIDEO_FEAT_DIM), dtype=np.float32))
        return False