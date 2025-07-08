import os
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import logging
import warnings

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TEXT_FEAT_DIM
from utils import get_logger

text_extractor_logger = get_logger('TextExtractor')

class RobustTextFeatureExtractor:
    def __init__(self, model_name='roberta-base', device=None):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if self.model.config.hidden_size != TEXT_FEAT_DIM:
            text_extractor_logger.warning(
                f"RoBERTa hidden size ({self.model.config.hidden_size}) does not match "
                f"TEXT_FEAT_DIM in config ({TEXT_FEAT_DIM}). "
            )
        text_extractor_logger.info(f"RoBERTa model loaded on {self.device}.")

    def extract(self, text):
        if not text.strip():
            return np.zeros(TEXT_FEAT_DIM, dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            encoded_input = self.tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )
        
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            output = self.model(**encoded_input)
        
        features = output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        
        if features.shape[0] != TEXT_FEAT_DIM:
            text_extractor_logger.error(
                f"Extracted feature dimension ({features.shape[0]}) does not match "
                f"TEXT_FEAT_DIM ({TEXT_FEAT_DIM})."
            )
            if features.shape[0] < TEXT_FEAT_DIM:
                temp_features = np.zeros(TEXT_FEAT_DIM, dtype=np.float32)
                temp_features[:features.shape[0]] = features
                features = temp_features
            else:
                features = features[:TEXT_FEAT_DIM]

        return features

def extract_text_features(transcript_path, output_dir, filename_base, extractor_instance):
    if not os.path.exists(transcript_path):
        text_extractor_logger.warning(f"Transcript file not found: {transcript_path}. Skipping.")
        return False

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"{filename_base}_roberta_features.npy")

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        features = extractor_instance.extract(text_content)
        np.save(output_filepath, features)
        return True
        
    except Exception as e:
        text_extractor_logger.error(f"Error extracting text features from {transcript_path}: {e}")
        np.save(output_filepath, np.zeros(TEXT_FEAT_DIM, dtype=np.float32))
        return False