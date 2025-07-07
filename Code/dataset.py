# Crime_Detection/Code/dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

# Adjust sys.path to import from sibling directory '..'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import FEATURE_PATHS, LABEL_MAPPING, SEQUENTIAL_AGGREGATION_METHOD, \
                   TEXT_FEAT_DIM, VIDEO_FEAT_DIM, AUDIO_FEAT_DIM

# Setup logger for the dataset (internal warnings/errors only)
dataset_logger = logging.getLogger('CrimeDataset')
if not dataset_logger.handlers:
    dataset_logger.addHandler(logging.NullHandler()) # Prevent "No handlers could be found"
dataset_logger.setLevel(logging.INFO) # Set to WARNING or ERROR for less verbosity during normal operation


class CrimeDataset(Dataset):
    """
    Custom Dataset for multimodal crime detection.
    Loads pre-extracted text, audio, and video features.
    """
    def __init__(self, video_sub_paths):
        """
        Args:
            video_sub_paths (list): List of relative video paths (e.g., '1-1004/clip_name.mp4')
                                    to include in this dataset instance.
        """
        self.aggregation_method = SEQUENTIAL_AGGREGATION_METHOD.lower()

        if not video_sub_paths:
            dataset_logger.error("video_sub_paths must be provided for dataset initialization.")
            raise ValueError("video_sub_paths cannot be empty.")
        
        self.data_items = self._load_data_items(video_sub_paths)
        dataset_logger.info(f"Initialized dataset with {len(self.data_items)} samples.")

    def _load_data_items(self, video_sub_paths):
        """
        Loads the file paths and labels for each data item.
        A data item corresponds to a single video clip/segment.
        """
        data_items = []
        dataset_logger.info(f"Attempting to load {len(video_sub_paths)} potential data items...")
        
        for rel_video_path in video_sub_paths:
            try:
                # Extract filename base and range folder from relative video path
                filename_base_no_ext = os.path.splitext(os.path.basename(rel_video_path))[0]
                range_folder = os.path.dirname(rel_video_path) 
                if not range_folder:
                    range_folder = "" # For top-level files

                # Construct feature paths based on the project structure
                audio_feat_path = os.path.join(FEATURE_PATHS['audio'], range_folder, filename_base_no_ext, f"{filename_base_no_ext}_yamnet_embeddings.npy")
                video_feat_path = os.path.join(FEATURE_PATHS['video'], range_folder, filename_base_no_ext, f"{filename_base_no_ext}_all.npy")
                text_feat_path = os.path.join(FEATURE_PATHS['text'], range_folder, f"{filename_base_no_ext}_roberta_features.npy")

                # Check if all feature files exist
                if not (os.path.exists(audio_feat_path) and
                        os.path.exists(video_feat_path) and
                        os.path.exists(text_feat_path)):
                    dataset_logger.warning(f"Skipping {rel_video_path}: One or more feature files missing. "
                                          f"Audio: {os.path.exists(audio_feat_path)}, "
                                          f"Video: {os.path.exists(video_feat_path)}, "
                                          f"Text: {os.path.exists(text_feat_path)}")
                    continue

                label = self._get_label_from_filename(filename_base_no_ext)

                data_items.append({
                    'id': filename_base_no_ext,
                    'audio_feat_path': audio_feat_path,
                    'video_feat_path': video_feat_path,
                    'text_feat_path': text_feat_path,
                    'label': label
                })
            except Exception as e:
                dataset_logger.error(f"Error loading data item for {rel_video_path}: {e}")
                continue
        return data_items

    def _get_label_from_filename(self, filename_base_no_ext):
        """
        Extracts the numerical label based on the filename convention, using LABEL_MAPPING from config.
        """
        if '_label_A' in filename_base_no_ext:
            return LABEL_MAPPING['_label_A']
        else:
            return LABEL_MAPPING['_label_N'] 
            
    def _aggregate_sequential_features(self, features, expected_dim):
        """
        Aggregates sequential features (video/audio) into a single vector.
        Handles empty or malformed input by returning a zero vector.
        """
        if features is None or features.size == 0:
            return np.zeros(expected_dim, dtype=np.float32)
        
        if features.ndim == 1:
            if features.shape[0] == expected_dim: 
                return features
            else:
                dataset_logger.warning(f"1D feature array provided for aggregation, but not of expected final dim {expected_dim}. Shape: {features.shape}. Attempting to pad/truncate.")
                padded = np.zeros(expected_dim, dtype=np.float32)
                min_len = min(expected_dim, features.shape[0])
                padded[:min_len] = features[:min_len]
                return padded

        if features.ndim != 2:
            dataset_logger.error(f"Expected 2D array for sequential features, got {features.ndim}D with shape {features.shape}. Returning zero vector.")
            return np.zeros(expected_dim, dtype=np.float32)

        if features.shape[1] != expected_dim:
            dataset_logger.warning(f"Feature dimension mismatch during aggregation for input. "
                                  f"Expected single feature dim {expected_dim}, got {features.shape[1]}. "
                                  f"Attempting to handle by reshaping/slicing if possible.")
            if features.shape[1] > expected_dim:
                features = features[:, :expected_dim]
            elif features.shape[1] < expected_dim:
                padded_features = np.zeros((features.shape[0], expected_dim), dtype=np.float32)
                padded_features[:, :features.shape[1]] = features
                features = padded_features

        if self.aggregation_method == 'mean':
            return np.mean(features, axis=0)
        elif self.aggregation_method == 'max':
            return np.max(features, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        try:
            text_features = np.load(item['text_feat_path'])
            audio_features_seq = np.load(item['audio_feat_path'])
            video_features_seq = np.load(item['video_feat_path'])

            audio_features = self._aggregate_sequential_features(audio_features_seq, AUDIO_FEAT_DIM)
            video_features = self._aggregate_sequential_features(video_features_seq, VIDEO_FEAT_DIM)

            def ensure_dim_and_type(arr, expected_dim, name):
                if arr is None or arr.size == 0:
                    dataset_logger.warning(f"Empty or None array for {name} for {item['id']}. Returning zero vector.")
                    return np.zeros(expected_dim, dtype=np.float32)
                
                if arr.ndim == 0:
                    arr = np.array([arr]) 
                
                if arr.shape[0] != expected_dim:
                    dataset_logger.warning(f"{name} feature shape mismatch for {item['id']}: {arr.shape}. Expected ({expected_dim},). Attempting to pad/truncate.")
                    padded_arr = np.zeros(expected_dim, dtype=np.float32)
                    min_len = min(expected_dim, arr.shape[0])
                    padded_arr[:min_len] = arr[:min_len]
                    return padded_arr
                return arr.astype(np.float32)

            text_features = ensure_dim_and_type(text_features, TEXT_FEAT_DIM, 'Text')
            audio_features = ensure_dim_and_type(audio_features, AUDIO_FEAT_DIM, 'Audio')
            video_features = ensure_dim_and_type(video_features, VIDEO_FEAT_DIM, 'Video')

            text_features = torch.tensor(text_features, dtype=torch.float32)
            audio_features = torch.tensor(audio_features, dtype=torch.float32)
            video_features = torch.tensor(video_features, dtype=torch.float32)
            label = torch.tensor(item['label'], dtype=torch.long)

            sample = {
                'id': item['id'],
                'text_features': text_features,
                'audio_features': audio_features,
                'video_features': video_features,
                'label': label
            }

            return sample
        
        except Exception as e:
            dataset_logger.error(f"Critical error loading/processing item {item['id']} at index {idx}: {e}. Returning dummy sample.")
            dummy_sample = {
                'id': item['id'],
                'text_features': torch.zeros(TEXT_FEAT_DIM, dtype=torch.float32),
                'audio_features': torch.zeros(AUDIO_FEAT_DIM, dtype=torch.float32),
                'video_features': torch.zeros(VIDEO_FEAT_DIM, dtype=torch.float32),
                'label': torch.tensor(LABEL_MAPPING['_label_N'], dtype=torch.long)
            }
            return dummy_sample