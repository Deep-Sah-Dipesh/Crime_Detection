# Crime_Detection/Code/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Adjust sys.path to import from sibling directory '..'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TEXT_FEAT_DIM, VIDEO_FEAT_DIM, AUDIO_FEAT_DIM, \
                   HIDDEN_DIM_TEXT, HIDDEN_DIM_VIDEO, HIDDEN_DIM_AUDIO, \
                   FUSION_HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE

class MultimodalCrimeDetector(nn.Module):
    """
    Multimodal fusion model for crime detection.
    Combines text, video, and audio features through modality-specific embedders,
    a fusion layer, and a final classifier.
    """
    def __init__(self):
        super(MultimodalCrimeDetector, self).__init__()

        # Modality-specific embedding layers to project features to a common hidden dimension
        self.text_embedder = nn.Sequential(
            nn.Linear(TEXT_FEAT_DIM, HIDDEN_DIM_TEXT),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        self.video_embedder = nn.Sequential(
            nn.Linear(VIDEO_FEAT_DIM, HIDDEN_DIM_VIDEO),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        self.audio_embedder = nn.Sequential(
            nn.Linear(AUDIO_FEAT_DIM, HIDDEN_DIM_AUDIO),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Calculate the total input dimension for the fusion layer
        total_fused_input_dim = HIDDEN_DIM_TEXT + HIDDEN_DIM_VIDEO + HIDDEN_DIM_AUDIO

        # Fusion Layer: Concatenates embedded features and applies a shared linear layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_fused_input_dim, FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Decision Layer (Classifier)
        self.classifier = nn.Linear(FUSION_HIDDEN_DIM, NUM_CLASSES)

    def forward(self, text_features, video_features, audio_features):
        """
        Forward pass through the multimodal model.
        Args:
            text_features (torch.Tensor): Batch of text features (Batch_size, TEXT_FEAT_DIM).
            video_features (torch.Tensor): Batch of video features (Batch_size, VIDEO_FEAT_DIM).
            audio_features (torch.Tensor): Batch of audio features (Batch_size, AUDIO_FEAT_DIM).
        Returns:
            torch.Tensor: Logits for classification (Batch_size, NUM_CLASSES).
        """
        # Embed each modality
        embedded_text = self.text_embedder(text_features)
        embedded_video = self.video_embedder(video_features)
        embedded_audio = self.audio_embedder(audio_features)
        
        # Concatenate embedded features along the feature dimension (dim=1)
        fused_features = torch.cat((embedded_text, embedded_video, embedded_audio), dim=1)
        
        # Pass through fusion layer
        fused_features = self.fusion_layer(fused_features)
        
        # Get final logits
        logits = self.classifier(fused_features)
        
        return logits