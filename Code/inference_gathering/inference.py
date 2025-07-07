import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
import logging
from tqdm import tqdm
import pandas as pd
import sys

# Adjust sys.path for internal imports within the Code directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_inference import Config
from dataset_for_infer import InferenceDataset
from model import MultimodalCrimeDetector # CORRECTED: Changed from 'models' to 'model'
from utils import load_checkpoint

os.makedirs(Config.OUTPUT_DIRS['logs'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.OUTPUT_DIRS['logs'], "inference_log.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('inference_script')

def perform_inference():
    logger.info("Starting inference script.")

    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    data_splits_path = os.path.join(Config.BASE_PATHS['data_splits'], 'data_splits.json')
    if not os.path.exists(data_splits_path):
        logger.error(f"Data splits file not found at {data_splits_path}. Cannot perform inference.")
        print(json.dumps({"status": "error", "message": f"Data splits file not found at {data_splits_path}"}))
        sys.exit(1)

    logger.info(f"Loading data splits from {data_splits_path}")
    with open(data_splits_path, 'r') as f:
        data_splits = json.load(f)

    clips_to_infer = data_splits['train'] 
    logger.info(f"Found {len(clips_to_infer)} clips for inference.")

    logger.info("Initializing Dataset for inference...")
    inference_dataset = InferenceDataset(clips_to_infer)
    
    # Check if there are any valid clips to process
    if len(inference_dataset) == 0:
        logger.error("No valid clips found in dataset for inference. Please ensure feature_metadata.json is correctly populated and feature files exist.")
        print(json.dumps({"status": "error", "message": "No valid clips found for inference. Check feature_metadata.json and feature files."}))
        sys.exit(1)

    inference_dataloader = DataLoader(inference_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True) 

    logger.info(f"Inference data batches: {len(inference_dataloader)}")

    model = MultimodalCrimeDetector(
        text_feature_dim=Config.TEXT_FEAT_DIM,
        video_feature_dim=Config.VIDEO_FEAT_DIM,
        audio_feature_dim=Config.AUDIO_FEAT_DIM,
        embedding_dim_text=Config.HIDDEN_DIM_TEXT,
        embedding_dim_video=Config.HIDDEN_DIM_VIDEO,
        embedding_dim_audio=Config.HIDDEN_DIM_AUDIO,
        fusion_dim=Config.FUSION_HIDDEN_DIM,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Placeholder optimizer

    loaded_model, _, _, _ = load_checkpoint(
        model, optimizer, Config.MODEL_SAVE_PATH, Config.CHECKPOINT_FILENAME
    )
    if loaded_model is None:
        logger.error(f"Failed to load model checkpoint from {os.path.join(Config.MODEL_SAVE_PATH, Config.CHECKPOINT_FILENAME)}. Cannot perform inference.")
        print(json.dumps({"status": "error", "message": "Failed to load model checkpoint."}))
        sys.exit(1)
    model = loaded_model
    logger.info(f"Successfully loaded model from {os.path.join(Config.MODEL_SAVE_PATH, Config.CHECKPOINT_FILENAME)}")

    model.eval() 
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_clip_ids = []
    total_loss = 0.0

    logger.info("Starting inference on clips...")
    with torch.no_grad():
        for batch in tqdm(inference_dataloader, desc="Performing Inference"):
            text_features = batch['text_features'].to(device)
            video_features = batch['video_features'].to(device)
            audio_features = batch['audio_features'].to(device)
            labels = batch['label'].to(device)
            clip_ids = batch['clip_id']

            outputs = model(text_features, video_features, audio_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_clip_ids.extend(clip_ids)

    avg_loss = total_loss / len(inference_dataloader)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision_crime = precision_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    summary_metrics = {
        'status': 'success',
        'Inference_Loss': avg_loss,
        'Accuracy': accuracy,
        'Precision_Crime_Class': precision_crime,
        'Recall_Crime_Class': recall,
        'F1_Score_Crime_Class': f1,
    }

    logger.info("--- Inference Summary ---")
    for key, value in summary_metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        elif key != 'status':
            logger.info(f"{key}: {value}")

    predictions_df = pd.DataFrame({
        'Clip_ID': all_clip_ids,
        'Predicted_Class': all_preds,
        'Actual_Class': all_labels
    })

    reverse_class_map = {v: k for k, v in Config.LABEL_MAPPING.items()}
    predictions_df['Predicted_Label'] = predictions_df['Predicted_Class'].map(reverse_class_map)
    predictions_df['Actual_Label'] = predictions_df['Actual_Class'].map(reverse_class_map)

    os.makedirs(Config.INFERENCE_OUTPUT_PATH, exist_ok=True)
    csv_filename = os.path.join(Config.INFERENCE_OUTPUT_PATH, "inference_predictions.csv")
    predictions_df.to_csv(csv_filename, index=False)
    logger.info(f"Detailed predictions saved to: {csv_filename}")

    summary_metrics['csv_output_path'] = csv_filename

    print(json.dumps(summary_metrics))

if __name__ == "__main__":
    try:
        perform_inference()
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        print(json.dumps({"status": "error", "message": f"An unhandled error occurred: {str(e)}"}))
        sys.exit(1)