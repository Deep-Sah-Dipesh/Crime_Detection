# Crime_Detection/Code/feature_extraction/audio_extractor.py

import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import shutil
import tempfile
import logging
import warnings

# Adjust sys.path to import from sibling directory '..'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import AUDIO_FEAT_DIM, BASE_PATHS # Import specific dimension and BASE_PATHS

# Setup logger for the audio extractor (internal warnings/errors only)
audio_extractor_logger = logging.getLogger('AudioExtractor')
if not audio_extractor_logger.handlers:
    audio_extractor_logger.addHandler(logging.NullHandler()) # Prevent "No handlers could be found"
audio_extractor_logger.setLevel(logging.INFO) # Set to WARNING or ERROR for less verbosity


class AudioFeatureExtractor:
    """
    YAMNet-based audio feature extractor with multiple fallback strategies.
    """
    
    def __init__(self, cache_dir=None):
        self.sample_rate = 16000
        self.model = None
        self.class_names = None
        self.cache_dir = cache_dir or os.path.join(BASE_PATHS['processed_data'], 'tfhub_cache')
        
        # Set persistent cache directory for TensorFlow Hub
        os.environ['TFHUB_CACHE_DIR'] = self.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._load_model_with_fallback() # Load model with fallback strategies
    
    def _load_model_with_fallback(self):
        """Load YAMNet with multiple fallback strategies."""
        strategies = [
            self._load_from_hub,
            self._load_after_cache_clear,
            self._load_with_retry
        ]
        
        for i, strategy in enumerate(strategies, 1):
            try:
                audio_extractor_logger.info(f"Attempting loading strategy {i}...")
                if strategy():
                    audio_extractor_logger.info(f"✅ Successfully loaded YAMNet using strategy {i}")
                    return
            except Exception as e:
                audio_extractor_logger.error(f"❌ Strategy {i} failed: {e}")
                continue
        
        raise RuntimeError("All YAMNet loading strategies failed")
    
    def _load_from_hub(self):
        """Strategy 1: Load directly from TensorFlow Hub."""
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self._load_class_names()
        return True
    
    def _load_after_cache_clear(self):
        """Strategy 2: Clear cache and reload."""
        default_cache = os.path.join(tempfile.gettempdir(), 'tfhub_modules')
        if os.path.exists(default_cache):
            shutil.rmtree(default_cache)
            audio_extractor_logger.info(f"Cleared default TFHub cache: {default_cache}")
        
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            audio_extractor_logger.info(f"Cleared custom TFHub cache: {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
        
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self._load_class_names()
        return True
    
    def _load_with_retry(self):
        """Strategy 3: Multiple retry attempts."""
        for attempt in range(3):
            try:
                audio_extractor_logger.info(f"  Retry attempt {attempt + 1}/3")
                self.model = hub.load('https://tfhub.dev/google/yamnet/1')
                self._load_class_names()
                return True
            except Exception as e:
                if attempt == 2:
                    raise e
                audio_extractor_logger.warning(f"  Attempt {attempt + 1} failed, retrying...")
                continue
        return False
    
    def _load_class_names(self):
        """Load YAMNet class names with error handling."""
        try:
            class_map_path = self.model.class_map_path().numpy().decode('utf-8')
            class_map_csv = pd.read_csv(class_map_path)
            self.class_names = class_map_csv['display_name'].tolist()
            audio_extractor_logger.info(f"✅ Loaded {len(self.class_names)} class names")
        except Exception as e:
            audio_extractor_logger.warning(f"Warning: Could not load class names: {e}")
            self.class_names = [f"class_{i}" for i in range(521)] # YAMNet has 521 classes
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio file for YAMNet."""
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            waveform = waveform.astype(np.float32)
            return waveform
        except Exception as e:
            audio_extractor_logger.error(f"❌ Error preprocessing audio {audio_path}: {e}")
            return None
    
    def extract(self, waveform):
        """
        Extract YAMNet embeddings from preprocessed waveform.
        YAMNet outputs (N, 1024) embeddings.
        """
        try:
            if not isinstance(waveform, tf.Tensor):
                waveform = tf.constant(waveform, dtype=tf.float32)
            
            scores, embeddings, spectrogram = self.model(waveform) 
            
            if embeddings.shape[-1] != AUDIO_FEAT_DIM:
                audio_extractor_logger.warning(
                    f"YAMNet embeddings dimension mismatch: Expected {AUDIO_FEAT_DIM}, Got {embeddings.shape[-1]}. "
                    f"This might indicate an issue with the YAMNet model or config. "
                    f"Features will be returned as is, potentially causing issues downstream."
                )
            
            return {
                'embeddings': embeddings.numpy(), # (N, 1024)
                'scores': scores.numpy(),
                'spectrogram': spectrogram.numpy(),
                'class_names': self.class_names
            }
        except Exception as e:
            audio_extractor_logger.error(f"❌ Error extracting features: {e}")
            return None
    
    def get_top_classes(self, scores, top_k=5):
        """Get top-k predicted classes for each audio segment."""
        top_classes_info = []
        for i, segment_scores in enumerate(scores):
            top_indices = np.argsort(segment_scores)[-top_k:][::-1]
            segment_info = {
                'segment_idx': i,
                'time_start': i * 0.48,  # YAMNet frame rate (fixed by model)
                'top_classes': []
            }
            for rank, class_idx in enumerate(top_indices):
                class_info = {
                    'rank': rank + 1,
                    'class_name': self.class_names[class_idx],
                    'score': float(segment_scores[class_idx])
                }
                segment_info['top_classes'].append(class_info)
            top_classes_info.append(segment_info)
        return top_classes_info
    
    def save_features(self, features, output_dir, filename_base):
        """Save extracted features to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings_path = os.path.join(output_dir, f"{filename_base}_yamnet_embeddings.npy")
        np.save(embeddings_path, features['embeddings'])
        
        top_classes = self.get_top_classes(features['scores'])
        csv_data = []
        for segment in top_classes:
            for class_info in segment['top_classes']:
                csv_data.append({
                    'segment_index': segment['segment_idx'],
                    'time_start_sec': segment['time_start'],
                    'rank': class_info['rank'],
                    'class_name': class_info['class_name'],
                    'score': class_info['score']
                })
        
        csv_path = None 
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(output_dir, f"{filename_base}_yamnet_classes.csv")
            df.to_csv(csv_path, index=False)
        else:
            audio_extractor_logger.warning(f"No class prediction data to save for {filename_base}. CSV not created.")
            
        spectrogram_path = os.path.join(output_dir, f"{filename_base}_yamnet_spectrogram.npy")
        np.save(spectrogram_path, features['spectrogram'])
        
        return {
            'embeddings_path': embeddings_path,
            'csv_path': csv_path,
            'spectrogram_path': spectrogram_path
        }

def extract_audio_features(audio_path, output_dir, filename_base, extractor_instance):
    """
    Extracts and saves audio features for a single file using the provided extractor instance.
    Returns True on success, False on failure.
    """
    if not os.path.exists(audio_path):
        audio_extractor_logger.warning(f"Audio file not found: {audio_path}. Skipping.")
        return False

    try:
        waveform = extractor_instance.preprocess_audio(audio_path)
        if waveform is None:
            audio_extractor_logger.error(f"Failed to preprocess audio: {audio_path}")
            return False
        
        features = extractor_instance.extract(waveform)
        if features is None:
            audio_extractor_logger.error(f"Failed to extract features from: {audio_path}")
            return False
        
        extractor_instance.save_features(features, output_dir, filename_base)
        return True
        
    except Exception as e:
        audio_extractor_logger.error(f"Error processing {audio_path}: {e}")
        return False