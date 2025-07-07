# Crime_Detection/Code/feature_extraction/text_extractor.py

import os
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import logging
import warnings # To suppress Hugging Face warnings

# Adjust sys.path to import from sibling directory '..'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import TEXT_FEAT_DIM # Import the specific dimension from config

# Setup logger for the text extractor (internal warnings/errors only)
text_extractor_logger = logging.getLogger('TextExtractor')
if not text_extractor_logger.handlers:
    text_extractor_logger.addHandler(logging.NullHandler()) # Prevent "No handlers could be found"
text_extractor_logger.setLevel(logging.INFO) # Set to WARNING or ERROR for less verbosity

class RobustTextFeatureExtractor:
    """
    RoBERTa-based text feature extractor.
    """
    def __init__(self, model_name='roberta-base', device=None):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

        # Verify output dimension matches config
        if self.model.config.hidden_size != TEXT_FEAT_DIM:
            text_extractor_logger.warning(
                f"RoBERTa model hidden size ({self.model.config.hidden_size}) "
                f"does not match TEXT_FEAT_DIM in config ({TEXT_FEAT_DIM}). "
                f"Features will be extracted but check your configuration if this is unintended."
            )

    @torch.no_grad()
    def extract(self, text_content):
        """
        Extracts RoBERTa [CLS] token embedding from a single text string.
        Args:
            text_content (str): The text content to process.
        Returns:
            np.ndarray: A 1D numpy array of the [CLS] token embedding.
                        Returns a zero vector if text content is empty or extraction fails.
        """
        if not text_content or not text_content.strip():
            text_extractor_logger.warning("Empty or whitespace-only text content provided. Returning zero vector.")
            return np.zeros(TEXT_FEAT_DIM, dtype=np.float32)

        try:
            # Suppress specific Hugging Face warnings if they are benign for your use case
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # You might want to filter specific warnings
                inputs = self.tokenizer(text_content, return_tensors='pt', 
                                        padding='max_length', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            # The [CLS] token embedding is at the first position
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            
            if cls_embedding.ndim == 0: # Handle case where squeeze makes it scalar
                cls_embedding = np.array([cls_embedding])

            # Ensure the output dimension matches TEXT_FEAT_DIM
            if cls_embedding.shape[0] != TEXT_FEAT_DIM:
                text_extractor_logger.error(
                    f"Extracted RoBERTa feature dimension mismatch for text: Expected {TEXT_FEAT_DIM}, Got {cls_embedding.shape[0]}. "
                    f"Padding/truncating to match config."
                )
                padded_embedding = np.zeros(TEXT_FEAT_DIM, dtype=np.float32)
                min_len = min(TEXT_FEAT_DIM, cls_embedding.shape[0])
                padded_embedding[:min_len] = cls_embedding[:min_len]
                return padded_embedding

            return cls_embedding.astype(np.float32)

        except Exception as e:
            text_extractor_logger.error(f"Error extracting text features: {e}. Returning zero vector.")
            return np.zeros(TEXT_FEAT_DIM, dtype=np.float32)

def extract_text_features(transcript_path, output_dir, filename_base, extractor_instance):
    """
    Extracts and saves text features for a single transcript file.
    Args:
        transcript_path (str): Full path to the input transcript (.txt) file.
        output_dir (str): Directory where the feature file will be saved.
                         (e.g., features/text/{range_folder})
        filename_base (str): Base filename (without extension) for the output feature file.
        extractor_instance (RobustTextFeatureExtractor): An initialized extractor instance.
    Returns:
        bool: True if extraction and saving was successful, False otherwise.
    """
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
        text_extractor_logger.debug(f"Saved text features for {filename_base} to {output_filepath}")
        return True
    except Exception as e:
        text_extractor_logger.error(f"Error processing {transcript_path}: {e}")
        return False