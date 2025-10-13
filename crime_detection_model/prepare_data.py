import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
import config

def create_ground_truth_csv():
    """Scans video filenames to create a master CSV with one-hot encoded labels."""
    video_data = []
    print(f"Scanning directory for video files: {config.VIDEO_DIR}")

    for root, _, files in os.walk(config.VIDEO_DIR):
        for filename in files:
            if ".mp4" in filename and "_label_" in filename:
                try:
                    relative_subdir = os.path.relpath(root, config.VIDEO_DIR)
                    label_part = filename.split('_label_')[1].split('.mp4')[0]
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
        raise RuntimeError("No video files with valid labels found.")

    df = pd.DataFrame(video_data)
    print(f"Found {len(df)} videos with labels.")

    mlb = MultiLabelBinarizer(classes=config.ALL_LABEL_CODES)
    encoded_labels = mlb.fit_transform(df['labels'])
    encoded_df = pd.DataFrame(encoded_labels, columns=mlb.classes_)

    final_df = pd.concat([df[['video_filename', 'subdirectory']], encoded_df], axis=1)
    
    final_df.to_csv(config.OUTPUT_CSV, index=False)
    print(f"Success! Ground truth data saved to: {config.OUTPUT_CSV}")
    print("\n--- Sample of Created Data ---")
    print(final_df.head())

if __name__ == "__main__":
    create_ground_truth_csv()

