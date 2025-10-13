import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import logging
logging.getLogger('tensorflow_hub').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow_hub')
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

import sys
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import argparse
import random

try:
    from predict import get_prediction
    import config
except ImportError:
    print("Error: Make sure predict.py and config.py are present.", file=sys.stderr)
    sys.exit(1)

def setup_logging(log_dir):
    """Initializes a logger to save run information to a file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"batch_predict_log_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger('crime_batch_predict')
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger, log_filepath

def run_batch_prediction(video_dir, output_csv_path, num_videos=None):
    """Runs predictions on videos and saves results, showing the top prediction."""
    logger.info(f"Starting batch prediction on directory: {video_dir}")
    
    all_video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not all_video_files:
        logger.error(f"No video files found in '{video_dir}'.")
        return

    if num_videos is not None and num_videos < len(all_video_files):
        logger.info(f"Randomly selecting {num_videos} videos to process from a total of {len(all_video_files)}.")
        video_files_to_process = random.sample(all_video_files, num_videos)
    else:
        logger.info(f"Processing all {len(all_video_files)} videos found.")
        video_files_to_process = all_video_files

    results = []
    
    for filename in tqdm(video_files_to_process, desc="Running Predictions", ncols=100):
        file_path = os.path.join(video_dir, filename)
        logger.info(f"--- Processing: {filename} ---")
        try:
            predicted_results = get_prediction(file_path)
            
            if not predicted_results:
                top_prediction = ("N/A", 0.0)
            else:
                top_prediction = max(predicted_results, key=lambda item: item[1])

            top_label_code, top_confidence = top_prediction
            top_label_name = config.LABEL_MAP.get(top_label_code, "Unknown")
            
            results.append({
                "filename": filename,
                "top_prediction": top_label_name,
                "confidence": f"{top_confidence:.2%}"
            })
            logger.info(f"Successfully processed {filename}. Top Prediction: {top_label_name}")

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}", exc_info=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    logger.info(f"✅ Batch prediction complete. Results saved to: {output_csv_path}")
    logger.info("\n--- Sample of Predictions ---\n" + results_df.head().to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch prediction on a directory of videos.")
    parser.add_argument("video_directory", type=str, help="Path to the directory containing videos.")
    args = parser.parse_args()
    
    # Set this to a number for a random subset, or None to process all files.
    NUM_VIDEOS_TO_PROCESS = None 

    logger, log_file = setup_logging(config.LOG_DIR)
    logger.info(f"Log file for this run: {log_file}")

    output_dir = os.path.join(config.PROJECT_ROOT, "batch_prediction_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"prediction_results_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
        
    run_batch_prediction(args.video_directory, output_path, num_videos=NUM_VIDEOS_TO_PROCESS)

