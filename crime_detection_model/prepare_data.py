import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
import config

def create_ground_truth_csv(video_dir, output_csv):
    """
    Scans a directory of video files to create a master CSV with one-hot encoded labels.
    """
    video_data = []
    print(f"Scanning directory for video files: {video_dir}")

    for root, _, files in os.walk(video_dir):
        for filename in files:
            if filename.lower().endswith(('.mp4', '.avi', 'mov')) and "_label_" in filename:
                try:
                    relative_subdir = os.path.relpath(root, video_dir)
                    if relative_subdir == '.':
                        relative_subdir = '' 

                    label_part = filename.split('_label_')[1].rsplit('.', 1)[0]
                    labels = label_part.split('-')
                    
                    cleaned_labels = [label for label in labels if label in config.ALL_LABEL_CODES]

                    if cleaned_labels:
                        video_data.append({
                            "video_filename": filename,
                            "subdirectory": relative_subdir,
                            "labels": cleaned_labels
                        })
                except IndexError:
                    print(f"  - Warning: Could not parse labels from: {filename}")
                    continue
    
    if not video_data:
        raise RuntimeError(f"No video files with valid labels found in '{video_dir}'.")

    df = pd.DataFrame(video_data)
    print(f"Found {len(df)} videos with valid labels.")

    mlb = MultiLabelBinarizer(classes=config.ALL_LABEL_CODES)
    encoded_labels = mlb.fit_transform(df['labels'])
    encoded_df = pd.DataFrame(encoded_labels, columns=mlb.classes_)

    final_df = pd.concat([df[['video_filename', 'subdirectory']], encoded_df], axis=1)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    
    print(f"\nSuccess! Ground truth data saved to: {output_csv}")
    print("\n--- Sample of Created Data ---")
    print(final_df.head())

if __name__ == "__main__":

    INPUT_DIRECTORY = r"H:\Crime_Detection\Testing\Videos"
    
    OUTPUT_CSV_PATH = r"H:\Crime_Detection\data\testing_ground_truth.csv"
    
    print("Starting data preparation...")
    create_ground_truth_csv(INPUT_DIRECTORY, OUTPUT_CSV_PATH)
    print("Data preparation complete.")

