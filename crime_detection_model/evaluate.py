import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import logging
logging.getLogger('tensorflow_hub').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import argparse
import pandas as pd
import numpy as np

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
    log_filename = f"evaluate_log_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger('crime_evaluate')
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

def parse_label_from_filename(filename):
    """Extracts ground truth labels from the video's filename."""
    try:
        base_name = os.path.splitext(filename)[0]
        label_part = base_name.split('_label_')[1]
        
        # Filter out placeholders like '0'
        labels = [l for l in label_part.split('-') if l in config.ALL_LABEL_CODES]
        return labels if labels else None
    except IndexError:
        return None

def evaluate_model(test_dir, num_videos=None):
    """Runs evaluation on a directory of videos with labeled filenames."""
    logger.info(f"Starting evaluation on directory: {test_dir}")
    
    true_labels = []
    pred_labels_top1 = [] 
    pred_labels_top3 = [] 

    all_video_paths = []
    for root, _, files in os.walk(test_dir):
        for filename in files:
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                all_video_paths.append(os.path.join(root, filename))

    if not all_video_paths:
        logger.error(f"No video files found in '{test_dir}'.")
        return
        
    if num_videos is not None and num_videos < len(all_video_paths):
        logger.info(f"Randomly selecting {num_videos} videos to evaluate from a total of {len(all_video_paths)}.")
        video_files_to_process = random.sample(all_video_paths, num_videos)
    else:
        logger.info(f"Processing all {len(all_video_paths)} videos found.")
        video_files_to_process = all_video_paths

    try:
        for file_path in tqdm(video_files_to_process, desc="Evaluating Videos", ncols=100):
            filename = os.path.basename(file_path)
            true_label_list = parse_label_from_filename(filename)
            
            if not true_label_list:
                logger.warning(f"Could not parse valid labels from {filename}. Skipping.")
                continue
            
            predicted_results = get_prediction(file_path)
            
            top_1_code = sorted(predicted_results, key=lambda item: item[1], reverse=True)[0][0]
            top_3_codes = [code for code, prob in sorted(predicted_results, key=lambda item: item[1], reverse=True)[:3]]

            true_labels.append(true_label_list)
            pred_labels_top1.append([top_1_code])
            pred_labels_top3.append(top_3_codes)
    
    finally:
        logger.info(f"\n--- Process finished or interrupted. Processed {len(true_labels)} videos. ---")

    if not true_labels:
        logger.error("Evaluation could not be completed as no ground truth labels were parsed.")
        return

    # --- Metrics Calculation ---
    mlb = MultiLabelBinarizer(classes=config.ALL_LABEL_CODES)
    true_binarized = mlb.fit_transform(true_labels)
    pred_top1_binarized = mlb.transform(pred_labels_top1)
    target_names = [config.LABEL_MAP[code] for code in mlb.classes_]

    top3_match_count = sum(1 for true, top3_preds in zip(true_labels, pred_labels_top3) if not set(true).isdisjoint(set(top3_preds)))
    top3_partial_accuracy = top3_match_count / len(true_labels) if true_labels else 0
    strict_accuracy = accuracy_score(true_binarized, pred_top1_binarized)

    # --- Reporting ---
    report_header = "\n" + "="*70 + "\n" + " " * 22 + "MODEL EVALUATION REPORT" + "\n" + "="*70
    logger.info(report_header)
    
    accuracy_report = (
        "\n--- Accuracy Metrics ---\n"
        f"  - Top-3 Partial Accuracy: {top3_partial_accuracy:.2%}\n"
        f"    (Percentage of samples where any of the Top 3 predictions match a true crime)\n"
        f"  - Strict Accuracy (Top-1 Exact Match): {strict_accuracy:.2%}\n"
        f"    (Percentage of samples where the #1 prediction perfectly matches all true labels)\n"
    )
    logger.info(accuracy_report)
    
    report = classification_report(true_binarized, pred_top1_binarized, target_names=target_names, zero_division=0)
    logger.info("---\n\n" + report)

    # --- Plotting ---
    try:
        cm = np.array([list(np.sum(true_binarized, axis=0)), list(np.sum(pred_top1_binarized, axis=0))]).T
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted'], yticklabels=target_names)
        plt.title('Prediction Counts per Category (vs. Ground Truth)')
        plt.xlabel('Prediction Type')
        plt.ylabel('Crime Category')
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(config.PROJECT_ROOT), "evaluation_summary.png")
        plt.savefig(plot_path)
        logger.info(f"\nEvaluation summary plot saved to: {plot_path}")
    except Exception as e:
        logger.error(f"Could not generate evaluation plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the crime detection model.")
    parser.add_argument("test_directory", type=str, help="Path to the directory with labeled test videos.")
    args = parser.parse_args()
    
    NUM_VIDEOS_TO_EVALUATE = 25 

    logger, log_file = setup_logging(config.LOG_DIR)
    logger.info(f"Log file for this run: {log_file}")
        
    evaluate_model(args.test_directory, num_videos=NUM_VIDEOS_TO_EVALUATE)

