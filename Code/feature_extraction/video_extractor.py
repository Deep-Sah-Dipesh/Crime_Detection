# Crime_Detection/Code/feature_extraction/video_extractor.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import logging

# Adjust sys.path to import from sibling directory '..'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import VIDEO_FEAT_DIM # Import the specific dimension from config

# Setup logger for the video extractor (internal warnings/errors only)
video_extractor_logger = logging.getLogger('VideoExtractor')
if not video_extractor_logger.handlers:
    video_extractor_logger.addHandler(logging.NullHandler()) # Prevent "No handlers could be found"
video_extractor_logger.setLevel(logging.INFO) # Set to WARNING or ERROR for less verbosity


class VideoFeatureExtractor:
    """
    ResNet + LSTM based video feature extractor.
    Extracts features from video frames and encodes sequences using LSTM.
    """
    def __init__(self, frame_rate=1, sequence_length=16, device=None):
        self.frame_rate = frame_rate
        self.sequence_length = sequence_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cnn_model = self._load_cnn_model()
        self.lstm_model = self._load_lstm_model()
        self.transform = self._get_transform()
        
        video_extractor_logger.info(f"Video Feature Extractor initialized with frame_rate={frame_rate}, sequence_length={sequence_length}, device={self.device}")

    def _load_cnn_model(self):
        """Load pre-trained ResNet50 and modify for feature extraction."""
        try:
            model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            model.eval() # Set to evaluation mode
            model.to(self.device)
            video_extractor_logger.info("✅ ResNet50 CNN model loaded successfully.")
            return model
        except Exception as e:
            video_extractor_logger.error(f"❌ Error loading ResNet50 model: {e}")
            raise

    def _load_lstm_model(self):
        """
        Define a simple LSTM model to process sequence of CNN features.
        The output size should match the desired VIDEO_FEAT_DIM.
        """
        try:
            input_size = 2048 # ResNet50 features output (before flattening and passing to LSTM)
            hidden_size = 512 # LSTM hidden state
            num_layers = 1
            
            class SequenceEncoder(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size):
                    super(SequenceEncoder, self).__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size) # Map LSTM output to desired feature dim

                def forward(self, x):
                    # x shape: (batch_size, sequence_length, input_size)
                    lstm_out, (h_n, c_n) = self.lstm(x)
                    # Use the last hidden state for fixed-size sequence representation
                    # h_n shape: (num_layers, batch_size, hidden_size)
                    last_hidden_state = h_n[-1] # Take the last layer's hidden state
                    output = self.fc(last_hidden_state) # (batch_size, output_size)
                    return output
            
            model = SequenceEncoder(input_size, hidden_size, num_layers, VIDEO_FEAT_DIM)
            model.eval()
            model.to(self.device)
            video_extractor_logger.info("✅ LSTM sequence encoder loaded successfully.")
            return model
        except Exception as e:
            video_extractor_logger.error(f"❌ Error loading LSTM model: {e}")
            raise

    def _get_transform(self):
        """Get image transformations for ResNet50."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad() # No need to track gradients during feature extraction
    def extract(self, video_path):
        """Extracts CNN+LSTM features from a single video."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            frames = []
            frame_features = [] # Stores LSTM output for each sequence
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video or error

                # Process every 'frame_rate' frame
                if cap.get(cv2.CAP_PROP_POS_FRAMES) % self.frame_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    transformed_frame = self.transform(frame_rgb)
                    frames.append(transformed_frame)

                    if len(frames) == self.sequence_length:
                        batch_frames = torch.stack(frames).to(self.device) # (sequence_length, C, H, W)
                        
                        cnn_features = self.cnn_model(batch_frames) # (sequence_length, 2048, 1, 1)
                        cnn_features = cnn_features.squeeze().view(self.sequence_length, -1) # (sequence_length, 2048)
                        
                        lstm_output = self.lstm_model(cnn_features.unsqueeze(0)) 
                        frame_features.append(lstm_output.squeeze(0).cpu().numpy()) 

                        frames = [] # Reset for next sequence

            cap.release()
            
            if len(frames) > 0: # Process any remaining frames that didn't form a full sequence
                video_extractor_logger.warning(f"Video {video_path} has {len(frames)} trailing frames. Padding with zeros to form a full sequence.")
                num_padding = self.sequence_length - len(frames)
                padding_frame_shape = frames[0].shape 
                for _ in range(num_padding):
                    frames.append(torch.zeros(padding_frame_shape, dtype=frames[0].dtype)) 
                
                if len(frames) == self.sequence_length: # Re-check if padding made it a full seq
                    batch_frames = torch.stack(frames).to(self.device)
                    cnn_features = self.cnn_model(batch_frames)
                    cnn_features = cnn_features.squeeze().view(self.sequence_length, -1)
                    lstm_output = self.lstm_model(cnn_features.unsqueeze(0))
                    frame_features.append(lstm_output.squeeze(0).cpu().numpy())

            if not frame_features:
                video_extractor_logger.warning(f"No features extracted for {video_path}. Video might be too short or corrupted.")
                return None

            all_features = np.array(frame_features) # (Num_sequences, VIDEO_FEAT_DIM)
            return all_features

        except Exception as e:
            video_extractor_logger.error(f"❌ Error processing video {video_path}: {e}")
            return None


def extract_video_features(video_path, output_dir, filename_base, extractor_instance):
    """
    Extracts and saves video features for a single file using the provided extractor instance.
    Returns True on success, False on failure.
    """
    if not os.path.exists(video_path):
        video_extractor_logger.warning(f"Video file not found: {video_path}. Skipping.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"{filename_base}_all.npy") # The main feature file

    try:
        features = extractor_instance.extract(video_path)
        if features is None:
            video_extractor_logger.error(f"Failed to extract features from {video_path} (returned None).")
            return False

        np.save(output_filepath, features)
        video_extractor_logger.debug(f"Saved video features for {filename_base} to {output_filepath}")
        return True
    except Exception as e:
        video_extractor_logger.error(f"Error processing {video_path}: {e}")
        return False