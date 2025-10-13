import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import logging
logging.getLogger('tensorflow_hub').setLevel(logging.ERROR)

import sys
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import argparse

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

    # Use a unique name for the logger to avoid conflicts
    logger = logging.getLogger('crime_evaluation')
    logger.setLevel(logging.INFO)
    logger.propagate = False # Prevent logs from propagating to the root logger

    # Clear existing handlers to prevent duplicate logs in interactive sessions
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create and configure handlers
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
        # Handle complex filenames by taking the last _label_ part
        label_part = base_name.split('_label_')[-1]
        # Split by '-' to get individual crime codes
        return [code for code in label_part.split('-') if code in config.ALL_LABEL_CODES]
    except IndexError:
        return None

def evaluate_model(test_dir, num_videos=None):
    """Runs evaluation on a directory of videos with labeled filenames."""
    logger.info(f"Starting evaluation on directory: {test_dir}")
    
    true_labels = []
    pred_labels = []
    
    # Recursively find all valid labeled video files
    all_video_files = []
    for root, _, files in os.walk(test_dir):
        for filename in files:
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and '_label_' in filename:
                all_video_files.append(os.path.join(root, filename))

    if not all_video_files:
        logger.error("No valid labeled video files found in the specified directory or its subdirectories.")
        return

    # Select a random subset if num_videos is specified
    if num_videos is not None and num_videos < len(all_video_files):
        logger.info(f"Randomly selecting {num_videos} videos to evaluate from a total of {len(all_video_files)}.")
        video_files_to_process = random.sample(all_video_files, num_videos)
    else:
        logger.info(f"Evaluating all {len(all_video_files)} labeled videos found.")
        video_files_to_process = all_video_files

    for file_path in tqdm(video_files_to_process, desc="Evaluating Videos", ncols=100):
        filename = os.path.basename(file_path)
        true_label_list = parse_label_from_filename(filename)
        
        # Skip files where labels could not be parsed
        if not true_label_list:
            logger.warning(f"Could not parse valid labels from {filename}. Skipping.")
            continue
        
        predicted_results = get_prediction(file_path)
        # Get predictions that are above the 0.5 confidence threshold
        pred_label_list = [code for code, prob in predicted_results if prob >= 0.5]
        
        # If the model predicts nothing, default to 'Normal' (code 'A')
        if not pred_label_list:
            pred_label_list = ['A']

        true_labels.append(true_label_list)
        pred_labels.append(pred_label_list)

    if not true_labels:
        logger.error("Evaluation could not be completed as no ground truth labels were successfully parsed.")
        return

    # --- Metrics Calculation ---
    mlb = MultiLabelBinarizer(classes=config.ALL_LABEL_CODES)
    true_binarized = mlb.fit_transform(true_labels)
    pred_binarized = mlb.transform(pred_labels)
    target_names = [config.LABEL_MAP[code] for code in mlb.classes_]

    # --- Reporting ---
    report_header = "\n" + "="*70 + "\n" + " " * 22 + "MODEL EVALUATION REPORT" + "\n" + "="*70 + "\n"
    logger.info(report_header)

    # Calculate Partial Accuracy (Did we find all true crimes?)
    partial_match_count = sum(1 for i in range(len(true_labels)) if set(true_labels[i]).issubset(set(pred_labels[i])))
    partial_accuracy = partial_match_count / len(true_labels) if true_labels else 0
    strict_accuracy = accuracy_score(true_binarized, pred_binarized)
    
    accuracy_report = (
        "--- Accuracy Metrics ---\n"
        f"  - Partial Accuracy (Recall Focused): {partial_accuracy:.2%}\n"
        f"    (Percentage of samples where all true crimes were correctly detected, ignoring extra predictions)\n"
        f"  - Strict Accuracy (Exact Match Ratio): {strict_accuracy:.2%}\n"
        f"    (Percentage of samples where predicted labels perfectly match true labels)\n"
    )
    logger.info(accuracy_report)

    # Per-Class Performance Report
    report = classification_report(true_binarized, pred_binarized, target_names=target_names, zero_division=0)
    class_performance = "--- Performance by Crime Category (Precision, Recall, F1-Score) ---\n" + report
    logger.info(class_performance)

    # --- Visualization ---
    try:
        from sklearn.metrics import multilabel_confusion_matrix
        cm = multilabel_confusion_matrix(true_binarized, pred_binarized, labels=mlb.classes_)
        
        num_labels = len(target_names)
        fig, axes = plt.subplots(1, num_labels, figsize=(5 * num_labels, 4.5))
        # Handle case where there is only one label
        if num_labels == 1:
            axes = [axes]
        fig.suptitle('Confusion Matrices per Category', fontsize=20)
        
        for i, label_name in enumerate(target_names):
            sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'],
                        annot_kws={"size": 14})
            axes[i].set_title(label_name, fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        
        plot_path = os.path.join(config.PROJECT_ROOT, "evaluation_confusion_matrix.png")
        plt.savefig(plot_path)
        logger.info(f"\n--- Confusion Matrix plot saved to: {plot_path} ---")
        # plt.show() # Commented out to prevent blocking in non-interactive environments
    except Exception as e:
        logger.error(f"Could not generate confusion matrix plot. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the crime detection model on a directory of labeled videos.")
    parser.add_argument("test_directory", type=str, help="Path to the directory containing labeled videos.")
    args = parser.parse_args()

    # Set the number of videos to evaluate. Use None to evaluate all.
    NUM_VIDEOS_TO_EVALUATE = 25 
    
    logger, log_file = setup_logging(config.LOG_DIR)
    logger.info(f"Log file for this run: {log_file}")
    
    evaluate_model(args.test_directory, num_videos=NUM_VIDEOS_TO_EVALUATE)

