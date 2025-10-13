import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import logging
logging.getLogger('tensorflow_hub').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

    # Configure logger
    logger = logging.getLogger('crime_batch_predict')
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger, log_filepath

def save_results(results_list, output_path):
    """Saves the current list of results to a CSV file."""
    if not results_list: return
    df = pd.DataFrame(results_list)
    df.to_csv(output_path, index=False)

def run_batch_prediction(video_dir, output_csv_path, num_videos=None):
    """Runs predictions on videos and saves the top 3 results to a CSV."""
    logger.info(f"Starting batch prediction on directory: {video_dir}")
    
    all_video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not all_video_files:
        logger.error(f"No video files found in '{video_dir}'.")
        return

    # Select a random subset if num_videos is specified
    if num_videos is not None and num_videos < len(all_video_files):
        logger.info(f"Randomly selecting {num_videos} videos to process from a total of {len(all_video_files)}.")
        video_files_to_process = random.sample(all_video_files, num_videos)
    else:
        logger.info(f"Processing all {len(all_video_files)} videos found.")
        video_files_to_process = all_video_files

    results = []
    
    try:
        for i, filename in enumerate(tqdm(video_files_to_process, desc="Running Predictions", ncols=100)):
            file_path = os.path.join(video_dir, filename)
            logger.info(f"--- Processing: {filename} ---")
            try:
                predicted_results = get_prediction(file_path)
                
                top_3 = sorted(predicted_results, key=lambda item: item[1], reverse=True)[:3]

                row = {"filename": filename}
                for j, (code, prob) in enumerate(top_3):
                    row[f"prediction_{j+1}"] = config.LABEL_MAP.get(code, "Unknown")
                    row[f"confidence_{j+1}"] = f"{prob:.2%}"
                
                results.append(row)
                logger.info(f"Successfully processed {filename}. Top Prediction: {row.get('prediction_1', 'N/A')}")

                # Intermittently save progress
                if (i + 1) % 50 == 0:
                    logger.info(f"\n--- Intermittently saving results at video {i+1}/{len(video_files_to_process)} ---\n")
                    save_results(results, output_csv_path)

            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}", exc_info=True)
    
    finally:
        logger.info("\n--- Saving final results ---")
        save_results(results, output_csv_path)
        logger.info(f"✅ Batch prediction finished. Final results saved to: {output_csv_path}")
        
        # --- ENHANCED FINAL SUMMARY ---
        if results:
            results_df = pd.DataFrame(results)
            if 'confidence_1' in results_df.columns:
                results_df['conf_numeric'] = results_df['confidence_1'].str.rstrip('%').astype(float)
                sample_df = results_df.sort_values(by='conf_numeric', ascending=False).head(10)
                display_columns = [col for col in sample_df.columns if col != 'conf_numeric']
                logger.info("\n--- Sample of 10 Highest Confidence Predictions ---\n" + sample_df[display_columns].to_string())
            else:
                logger.info("\n--- Sample of Predictions ---\n" + results_df.head().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch prediction on a directory of videos.")
    parser.add_argument("video_directory", type=str, help="Path to the directory containing videos.")
    args = parser.parse_args()
    
    NUM_VIDEOS_TO_PROCESS = 10 

    logger, log_file = setup_logging(config.LOG_DIR)
    logger.info(f"Log file for this run: {log_file}")

    output_dir = os.path.join(config.PROJECT_ROOT, "batch_prediction_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"prediction_results_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
        
    run_batch_prediction(args.video_directory, output_path, num_videos=NUM_VIDEOS_TO_PROCESS)

