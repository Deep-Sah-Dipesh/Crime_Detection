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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import AUDIO_FEAT_DIM, BASE_PATHS
from utils import get_logger

audio_extractor_logger = get_logger('AudioExtractor')

class AudioFeatureExtractor:
    def __init__(self, cache_dir=None):
        self.sample_rate = 16000
        self.model = None
        self.class_names = None
        self.cache_dir = cache_dir or os.path.join(BASE_PATHS['processed_data'], 'tfhub_cache')
        
        os.environ['TFHUB_CACHE_DIR'] = self.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._load_model()

    def _load_model(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.model = hub.load('https://tfhub.dev/google/yamnet/1')
            self.class_names = self._load_yamnet_class_names()
            audio_extractor_logger.info("YAMNet model loaded successfully.")
        except Exception as e:
            audio_extractor_logger.error(f"Failed to load YAMNet model: {e}")
            self.model = None

    def _load_yamnet_class_names(self):
        try:
            return [
                "Speech", "Female speech", "Male speech", "Singing", "Choir", "Whispering", "Screaming", "Speech synthesizer", "Shout",
                "Frying (food)", "Boiling", "Blender", "Buzzer", "Alarm clock", "Whistle", "Burping", "Flatulence", "Gurgling",
                "Growling", "Purr", "Chirp", "Meow", "Bark", "Coo", "Roar", "Gasp", "Sigh", "Sniff", "Panting", "Yelp", "Howl",
                "Caw", "Bleat", "Oink", "Neigh", "Moo", "Quack", "Goat", "Frog", "Owl", "Cricket", "Cicada", "Mosquito", "Fly",
                "Bee", "Chainsaw", "Drill", "Jackhammer", "Hammer", "Sawing", "Grinding", "Cutting", "Filing", "Crushing",
                "Stroking", "Slam", "Punch", "Gunshot", "Explosion", "Firecracker", "Artillery fire", "Bomb", "Car crash",
                "Crash", "Impact", "Shatter", "Smash", "Breaking", "Tearing", "Scraping", "Scratch", "Rattle", "Clink", "Whir",
                "Squeak", "Creak", "Click", "Knock", "Thump", "Chime", "Gong", "Bell", "Jingle", "Ringtone", "Ding", "Clang",
                "Hiss", "Sizzle", "Fizz", "Splash", "Whoosh", "Swish", "Whirr", "Gurgle", "Bubble", "Drip", "Trickle", "Pour",
                "Flush", "Squeal", "Honk", "Beep", "Chug", "Chuff", "Hoot", "Trumpet", "Flute", "Clarinet", "Saxophone", "Oboe",
                "Bassoon", "Harmonica", "Accordion", "Bagpipes", "Tuba", "Trombone", "French horn", "Trumpet", "Guitar",
                "Piano", "Organ", "Harpsichord", "Violin", "Cello", "Double bass", "Harp", "Drum", "Tabla", "Conga", "Bongo",
                "Tambourine", "Cymbal", "Glockenspiel", "Xylophone", "Marimba", "Vibraphone", "Chimes", "Cowbell", "Triangle",
                "Cello", "Double bass", "Harp", "Drum", "Tabla", "Conga", "Bongo", "Tambourine", "Cymbal", "Glockenspiel",
                "Xylophone", "Marimba", "Vibraphone", "Chimes", "Cowbell", "Triangle"
            ]
        except Exception as e:
            audio_extractor_logger.warning(f"Could not load YAMNet class names: {e}. Proceeding without them.")
            return None

    def preprocess_audio(self, audio_path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                y, sr = librosa.load(audio_path, sr=self.sample_rate)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            return y
        except Exception as e:
            audio_extractor_logger.error(f"Error loading or resampling audio {audio_path}: {e}")
            return None

    def extract(self, waveform):
        if self.model is None:
            audio_extractor_logger.error("YAMNet model not loaded. Cannot extract features.")
            return None
        try:
            waveform_tensor = tf.constant(waveform, dtype=tf.float32)
            scores, embeddings, spectrogram = self.model(waveform_tensor)
            
            embeddings_np = embeddings.numpy()
            scores_np = scores.numpy()
            spectrogram_np = spectrogram.numpy()

            return {
                'embeddings': embeddings_np,
                'scores': scores_np,
                'spectrogram': spectrogram_np
            }
        except Exception as e:
            audio_extractor_logger.error(f"Error during YAMNet feature extraction: {e}")
            return None

    def save_features(self, features, output_dir, filename_base):
        os.makedirs(output_dir, exist_ok=True)

        embeddings_path = os.path.join(output_dir, f"{filename_base}_yamnet_embeddings.npy")
        np.save(embeddings_path, features['embeddings'])

        csv_path = None
        if self.class_names is not None:
            scores_df = pd.DataFrame(features['scores'], columns=self.class_names)
            csv_path = os.path.join(output_dir, f"{filename_base}_yamnet_scores.csv")
            try:
                scores_df.to_csv(csv_path, index=False)
            except Exception as e:
                audio_extractor_logger.warning(f"Failed to save class prediction data to save for {filename_base}. CSV not created.")
            
        spectrogram_path = os.path.join(output_dir, f"{filename_base}_yamnet_spectrogram.npy")
        np.save(spectrogram_path, features['spectrogram'])
        
        return {
            'embeddings_path': embeddings_path,
            'csv_path': csv_path,
            'spectrogram_path': spectrogram_path
        }

def extract_audio_features(audio_path, output_dir, filename_base, extractor_instance):
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
        audio_extractor_logger.error(f"Error during audio feature extraction for {audio_path}: {e}")
        return False