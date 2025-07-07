import torch
import numpy as np
from torch.utils.data import Dataset
import json
import os
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_inference import Config

logger = logging.getLogger('InferenceDataset')

class InferenceDataset(Dataset):
    def __init__(self, clip_ids):
        self.clip_ids = clip_ids
        self.aggregation_method = Config.SEQUENTIAL_AGGREGATION_METHOD.lower()
        self.feature_metadata = self._load_feature_metadata()
        
        self.data_items = self._prepare_data_items()
        
        logger.info(f"Initialized InferenceDataset with {len(self.data_items)} valid clips out of {len(self.clip_ids)} provided.")

    def _load_feature_metadata(self):
        try:
            with open(Config.FEATURE_METADATA_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Feature metadata file not found at {Config.FEATURE_METADATA_FILE}. Please ensure feature extraction was successful and this file was generated.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {Config.FEATURE_METADATA_FILE}. File might be corrupted.")
            return {}

    def _prepare_data_items(self):
        data_items = []
        for clip_id in self.clip_ids:
            if clip_id in self.feature_metadata:
                metadata = self.feature_metadata[clip_id]
                text_feat_path = metadata.get('text_feature_path')
                audio_feat_path = metadata.get('audio_feature_path')
                video_feat_path = metadata.get('video_feature_path')
                label = metadata.get('label')

                if text_feat_path and os.path.exists(text_feat_path) and \
                   audio_feat_path and os.path.exists(audio_feat_path) and \
                   video_feat_path and os.path.exists(video_feat_path) and \
                   label is not None:
                    data_items.append({
                        'id': clip_id,
                        'text_feat_path': text_feat_path,
                        'audio_feat_path': audio_feat_path,
                        'video_feat_path': video_feat_path,
                        'label': label
                    })
                else:
                    logger.warning(f"Skipping clip {clip_id}: One or more feature files missing on disk or label is missing. "
                                   f"Text: {os.path.exists(text_feat_path) if text_feat_path else 'N/A'}, "
                                   f"Audio: {os.path.exists(audio_feat_path) if audio_feat_path else 'N/A'}, "
                                   f"Video: {os.path.exists(video_feat_path) if video_feat_path else 'N/A'}, "
                                   f"Label: {label is not None}")
            else:
                logger.warning(f"Skipping clip {clip_id}: Not found in feature_metadata.json.")
        return data_items

    def _aggregate_sequential_features(self, features, expected_dim):
        if features is None or features.size == 0:
            return np.zeros(expected_dim, dtype=np.float32)
        
        if features.ndim == 1:
            if features.shape[0] == expected_dim: 
                return features
            else:
                padded = np.zeros(expected_dim, dtype=np.float32)
                min_len = min(expected_dim, features.shape[0])
                padded[:min_len] = features[:min_len]
                return padded
        
        if features.ndim != 2:
            logger.error(f"Expected 2D array for sequential features, got {features.ndim}D with shape {features.shape}. Returning zero vector.")
            return np.zeros(expected_dim, dtype=np.float32)

        if features.shape[1] != expected_dim:
            logger.warning(f"Feature dimension mismatch during aggregation for input. "
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
            text_features_raw = np.load(item['text_feat_path'])
            audio_features_seq_raw = np.load(item['audio_feat_path'])
            video_features_seq_raw = np.load(item['video_feat_path'])

            audio_features = self._aggregate_sequential_features(audio_features_seq_raw, Config.AUDIO_FEAT_DIM)
            video_features = self._aggregate_sequential_features(video_features_seq_raw, Config.VIDEO_FEAT_DIM)

            def ensure_dim_and_type(arr, expected_dim, name):
                if arr is None or arr.size == 0:
                    logger.warning(f"Empty or None array for {name} for {item['id']}. Returning zero vector.")
                    return np.zeros(expected_dim, dtype=np.float32)
                
                if arr.ndim == 0:
                    arr = np.array([arr]) 
                
                if arr.shape[0] != expected_dim:
                    logger.warning(f"{name} feature shape mismatch for {item['id']}: {arr.shape}. Expected ({expected_dim},). Attempting to pad/truncate.")
                    padded_arr = np.zeros(expected_dim, dtype=np.float32)
                    min_len = min(expected_dim, arr.shape[0])
                    padded_arr[:min_len] = arr[:min_len]
                    return padded_arr
                return arr.astype(np.float32)

            text_features = ensure_dim_and_type(text_features_raw, Config.TEXT_FEAT_DIM, 'Text')
            audio_features = ensure_dim_and_type(audio_features, Config.AUDIO_FEAT_DIM, 'Audio')
            video_features = ensure_dim_and_type(video_features, Config.VIDEO_FEAT_DIM, 'Video')

            text_features = torch.tensor(text_features, dtype=torch.float32)
            audio_features = torch.tensor(audio_features, dtype=torch.float32)
            video_features = torch.tensor(video_features, dtype=torch.float32)
            label = torch.tensor(item['label'], dtype=torch.long)

            sample = {
                'clip_id': item['id'],
                'text_features': text_features,
                'audio_features': audio_features,
                'video_features': video_features,
                'label': label
            }

            return sample
        
        except Exception as e:
            logger.error(f"Critical error loading/processing item {item['id']} at index {idx}: {e}. Returning dummy sample.")
            dummy_sample = {
                'clip_id': item['id'],
                'text_features': torch.zeros(Config.TEXT_FEAT_DIM, dtype=torch.float32),
                'audio_features': torch.zeros(Config.AUDIO_FEAT_DIM, dtype=torch.float32),
                'video_features': torch.zeros(Config.VIDEO_FEAT_DIM, dtype=torch.float32),
                'label': torch.tensor(Config.LABEL_MAPPING['_label_N'], dtype=torch.long)
            }
            return dummy_sample