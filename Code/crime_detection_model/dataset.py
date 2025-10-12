# dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import config

class CrimeFeatureDataset(Dataset):
    def __init__(self, annotations_csv):
        self.annotations = pd.read_csv(annotations_csv)
        # Labels are now all columns after the first two (filename, subdirectory)
        self.labels_columns = self.annotations.columns[2:] 

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_info = self.annotations.iloc[idx]
        base_filename = os.path.splitext(video_info['video_filename'])[0]
        subdirectory = video_info['subdirectory']

        # --- CORRECTED PATH LOGIC ---
        video_feat_path = os.path.join(config.VIDEO_FEATURES_DIR, subdirectory, base_filename, f"{base_filename}_all.npy")
        audio_feat_path = os.path.join(config.AUDIO_FEATURES_DIR, subdirectory, base_filename, f"{base_filename}_yamnet_embeddings.npy")
        text_feat_path = os.path.join(config.TEXT_FEATURES_DIR, subdirectory, f"{base_filename}_roberta_features.npy")

        try:
            # Load and aggregate features
            video_features_seq = np.load(video_feat_path)
            video_features = torch.from_numpy(video_features_seq.mean(axis=0)).float()

            audio_features_seq = np.load(audio_feat_path)
            audio_features = torch.from_numpy(audio_features_seq.mean(axis=0)).float()

            text_features = torch.from_numpy(np.load(text_feat_path)).float().squeeze()
        
        except FileNotFoundError as e:
            print(f"\nERROR: Feature file not found for '{base_filename}'.")
            print(f"Tried to access: {e.filename}")
            return None # We will filter these out in the DataLoader

        combined_features = torch.cat((video_features, audio_features, text_features), dim=-1)
        
        if combined_features.shape[0] != config.INPUT_FEATURE_SIZE:
             print(f"\nWARNING: Mismatched feature size for {base_filename}. "
                   f"Expected {config.INPUT_FEATURE_SIZE}, got {combined_features.shape[0]}. Skipping.")
             return None

        label_vector = torch.tensor(video_info[self.labels_columns].values.astype(float), dtype=torch.float32)

        return combined_features, label_vector

def collate_fn(batch):
    """
    Custom collate function to filter out None values from the batch
    that may result from FileNotFoundError in the dataset.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)