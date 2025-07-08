import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TEXT_FEAT_DIM, VIDEO_FEAT_DIM, AUDIO_FEAT_DIM, \
                   HIDDEN_DIM_TEXT, HIDDEN_DIM_VIDEO, HIDDEN_DIM_AUDIO, \
                   FUSION_HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE

class MultimodalCrimeDetector(nn.Module):
    def __init__(self):
        super(MultimodalCrimeDetector, self).__init__()

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
        
        total_fused_input_dim = HIDDEN_DIM_TEXT + HIDDEN_DIM_VIDEO + HIDDEN_DIM_AUDIO
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_fused_input_dim, FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        self.classifier = nn.Linear(FUSION_HIDDEN_DIM, NUM_CLASSES)

    def forward(self, text_features, video_features, audio_features):
        embedded_text = self.text_embedder(text_features)
        embedded_video = self.video_embedder(video_features)
        embedded_audio = self.audio_embedder(audio_features)
        
        fused_features = torch.cat((embedded_text, embedded_video, embedded_audio), dim=1)
        
        fused_features = self.fusion_layer(fused_features)
        
        logits = self.classifier(fused_features)
        return logits