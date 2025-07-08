import os
import torch
import numpy as np
from torch.utils.data import Dataset
import logging

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils import get_logger

dataset_logger = get_logger('CrimeDataset')

class CrimeDataset(Dataset):
    def __init__(self, video_sub_paths):
        self.aggregation_method = config.SEQUENTIAL_AGGREGATION_METHOD.lower()

        if not video_sub_paths:
            dataset_logger.warning("No video sub-paths provided to the dataset. Dataset will be empty.")
            self.data = []
            return

        self.data = self._prepare_data_list(video_sub_paths)
        dataset_logger.info(f"Initialized CrimeDataset with {len(self.data)} samples using {self.aggregation_method} aggregation.")

    def _prepare_data_list(self, video_sub_paths):
        prepared_data = []
        for video_sub_path in video_sub_paths:
            # video_sub_path example: '1-1004/A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A.mp4'
            clip_dir_relative = os.path.dirname(video_sub_path)
            clip_filename_base = os.path.splitext(os.path.basename(video_sub_path))[0]

            label = config.LABEL_MAPPING['_label_A'] if '_label_A' in clip_filename_base else config.LABEL_MAPPING['_label_N']

            # Construct full paths to feature files based on the expected structure
            # Video and Audio features are in a subfolder named after the clip_filename_base
            # Text features are in the range_folder directly
            text_feature_path = os.path.join(config.FEATURE_PATHS['text'], clip_dir_relative, f"{clip_filename_base}_roberta_features.npy")
            audio_feature_path = os.path.join(config.FEATURE_PATHS['audio'], clip_dir_relative, clip_filename_base, f"{clip_filename_base}_yamnet_embeddings.npy")
            video_feature_path = os.path.join(config.FEATURE_PATHS['video'], clip_dir_relative, clip_filename_base, f"{clip_filename_base}_all.npy")

            if not (os.path.exists(text_feature_path) and 
                    os.path.exists(audio_feature_path) and 
                    os.path.exists(video_feature_path)):
                dataset_logger.warning(f"Skipping {video_sub_path}: One or more feature files missing.")
                continue
            
            prepared_data.append({
                'id': clip_filename_base,
                'text_path': text_feature_path,
                'audio_path': audio_feature_path,
                'video_path': video_feature_path,
                'label': label
            })
        return prepared_data

    def __len__(self):
        return len(self.data)

    def _aggregate_sequential_features(self, features, method):
        if features is None or features.size == 0:
            return None

        if features.ndim == 1:
            return features

        if method == 'mean':
            return np.mean(features, axis=0)
        elif method == 'max':
            return np.max(features, axis=0)
        else:
            dataset_logger.error(f"Unsupported aggregation method: {method}. Defaulting to mean.")
            return np.mean(features, axis=0)

    def __getitem__(self, idx):
        item = self.data[idx]

        def load_npy_robust(path, default_dim, modality_name):
            try:
                features = np.load(path)
                if features.ndim > 1 and modality_name != 'Text':
                    features = self._aggregate_sequential_features(features, self.aggregation_method)
                
                if features.shape[0] != default_dim:
                    dataset_logger.error(f"Shape mismatch for {modality_name} features from {path}. Expected {default_dim}, got {features.shape[0]}. Returning zero vector.")
                    return np.zeros(default_dim, dtype=np.float32)
                return features.astype(np.float32)
            except FileNotFoundError:
                dataset_logger.warning(f"Feature file not found: {path}. Returning zero vector for {modality_name}.")
                return np.zeros(default_dim, dtype=np.float32)
            except Exception as e:
                dataset_logger.error(f"Error loading or processing {modality_name} feature file {path}: {e}. Returning zero vector.")
                return np.zeros(default_dim, dtype=np.float32)

        def ensure_dim_and_type(arr, expected_dim, modality_name):
            if arr is None:
                return np.zeros(expected_dim, dtype=np.float32)
            if arr.shape[0] != expected_dim:
                dataset_logger.error(f"Dimension mismatch for {modality_name} features after processing. Expected {expected_dim}, got {arr.shape[0]}. Returning zero vector.")
                return np.zeros(expected_dim, dtype=np.float32)
            return arr.astype(np.float32)

        try:
            text_features = load_npy_robust(item['text_path'], config.TEXT_FEAT_DIM, 'Text')
            audio_features = load_npy_robust(item['audio_path'], config.AUDIO_FEAT_DIM, 'Audio')
            video_features = load_npy_robust(item['video_path'], config.VIDEO_FEAT_DIM, 'Video')

            text_features = ensure_dim_and_type(text_features, config.TEXT_FEAT_DIM, 'Text')
            audio_features = ensure_dim_and_type(audio_features, config.AUDIO_FEAT_DIM, 'Audio')
            video_features = ensure_dim_and_type(video_features, config.VIDEO_FEAT_DIM, 'Video')

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
                'text_features': torch.zeros(config.TEXT_FEAT_DIM, dtype=torch.float32),
                'audio_features': torch.zeros(config.AUDIO_FEAT_DIM, dtype=torch.float32),
                'video_features': torch.zeros(config.VIDEO_FEAT_DIM, dtype=torch.float32),
                'label': torch.tensor(0, dtype=torch.long)
            }
            return dummy_sample