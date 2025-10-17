import os
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys

try:
    config_dir = r"H:\Crime_Detection\crime_detection_model"
    sys.path.insert(0, config_dir)
    import config
except ImportError:
    print(f"Error: Could not import config.py from '{config_dir}'.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()


def get_video_metadata(file_path):
    """Calculates the duration and frame count of a video file."""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 0.0, 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration, frame_count
    except Exception:
        return 0.0, 0

def get_word_count(transcript_path):
    """Calculates the number of words in a transcript file."""
    try:
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return len(f.read().split())
        return 0
    except Exception:
        return 0

def parse_labels_from_filename(filename):
    """Extracts ground truth labels from a filename."""
    try:
        label_part = filename.split('_label_')[1].rsplit('.', 1)[0]
        labels = [l for l in label_part.split('-') if l in config.ALL_LABEL_CODES]
        return labels if labels else []
    except IndexError:
        return []

def analyze_dataset_metadata(video_dir, output_report_path):
    """
    Scans a directory of videos to extract metadata, print a report,
    and save the report to a text file.
    """
    print(f"--- Starting Metadata Analysis for Directory: {os.path.basename(video_dir)} ---")
    
    training_root = os.path.abspath(os.path.join(video_dir, '..'))
    transcripts_dir = os.path.join(training_root, 'Processed_Data', 'Transcripts')
    print(f"Searching for transcripts in: {transcripts_dir}")
    
    video_metadata = []
    video_files = []
    for root, _, files in os.walk(video_dir):
        for filename in files:
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, filename))

    if not video_files:
        print(f"No video files found in '{video_dir}'.")
        return

    for file_path in tqdm(video_files, desc="Analyzing Videos", ncols=100):
        filename = os.path.basename(file_path)
        base_filename = os.path.splitext(filename)[0]
        
        relative_path_from_videos = os.path.relpath(file_path, video_dir)
        relative_subdir = os.path.dirname(relative_path_from_videos)
        
        duration, frame_count = get_video_metadata(file_path)
        labels = parse_labels_from_filename(filename)
        
        transcript_path = os.path.join(transcripts_dir, relative_subdir, f"{base_filename}.txt")
        word_count = get_word_count(transcript_path)
        
        if duration > 0:
            video_metadata.append({
                'duration_seconds': duration,
                'frame_count': frame_count,
                'word_count': word_count,
                'labels': labels,
                'label_count': len(labels)
            })

    if not video_metadata:
        print("Could not process any videos to generate metadata.")
        return

    df = pd.DataFrame(video_metadata)

    # --- Calculate Metrics ---
    total_videos = len(df)
    total_duration_hours = df['duration_seconds'].sum() / 3600
    num_categories = len(config.ALL_LABEL_CODES)
    avg_video_length_seconds = df['duration_seconds'].mean()
    avg_frames_per_video = df['frame_count'].mean()
    avg_words_per_transcript = df[df['word_count'] > 0]['word_count'].mean() if not df[df['word_count'] > 0].empty else 0
    avg_instances_per_video = df['label_count'].mean()
    all_labels = [label for sublist in df['labels'] for label in sublist]
    total_label_instances = len(all_labels)
    avg_instances_per_category = total_label_instances / num_categories if num_categories > 0 else 0
    min_video_length = df['duration_seconds'].min()
    max_video_length = df['duration_seconds'].max()

    # --- Generate Report String ---
    report = f"""
======================================================================
                  DATASET METADATA REPORT
======================================================================

Dataset Path: {video_dir}

--- Overall Statistics ---
Total Videos Analyzed:      {total_videos}
Total Duration:             {total_duration_hours:.2f} hours
Number of Categories:       {num_categories} (Including 'Normal')

--- Averages ---
Avg. Video Length:          {avg_video_length_seconds:.2f} seconds
Avg. Frames per Video:      {avg_frames_per_video:,.0f}
Avg. Words per Transcript:  {avg_words_per_transcript:.2f} (Calculated only for videos with transcripts)
Avg. Labels per Video:      {avg_instances_per_video:.2f}
Avg. Videos per Category:   {avg_instances_per_category:.2f}

--- Extremes ---
Min. Video Length:          {min_video_length:.2f} seconds
Max. Video Length:          {max_video_length:.2f} seconds

======================================================================
"""
    # Print report to console
    print(report)

    # Save report to a text file
    try:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n✅ Report successfully saved to: {output_report_path}")
    except Exception as e:
        print(f"\n❌ Error saving report to file: {e}")


if __name__ == "__main__":
    # --- CONFIGURE YOUR ANALYSIS HERE ---
    DATA_OUTPUT_DIR = r"H:\Crime_Detection\data"

    # # --- Option 1: Analyze your TRAINING dataset ---
    # DIRECTORY_TO_ANALYZE = r"H:\Crime_Detection\Training\Videos"
    # OUTPUT_REPORT_PATH = os.path.join(DATA_OUTPUT_DIR, "training_set_metadata.txt")
    
    # --- Option 2: Analyze your TESTING dataset ---
    DIRECTORY_TO_ANALYZE = r"H:\Crime_Detection\Testing\Videos"
    OUTPUT_REPORT_PATH = os.path.join(DATA_OUTPUT_DIR, "testing_set_metadata.txt")
    
    analyze_dataset_metadata(DIRECTORY_TO_ANALYZE, OUTPUT_REPORT_PATH)

