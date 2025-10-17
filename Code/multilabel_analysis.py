import pandas as pd
import os
from collections import Counter

GROUND_TRUTH_CSV = r"H:\Crime_Detection\data\testing_ground_truth.csv"

LABEL_MAP = {
    'A': 'Normal', 'B1': 'Fighting', 'B2': 'Shooting', 'B4': 'Riot',
    'B5': 'Abuse', 'B6': 'Car Accident', 'G': 'Explosion'
}

LABEL_ORDER = ['A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G']

def analyze_multilabel_distribution(csv_path):
    """
    Reads the ground truth CSV and analyzes the distribution and
    common combinations of multi-labeled videos.
    """
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    print("--- Analyzing Multi-Label Distribution ---")
    print(f"Source CSV: {csv_path}")
    print(f"Total Unique Videos Analyzed: {len(df)}")
    
    label_columns = [col for col in df.columns if col in LABEL_MAP]
    df['label_count'] = df[label_columns].sum(axis=1)

    label_count_distribution = df['label_count'].value_counts().sort_index()

    print("\n--- Distribution of Videos by Number of Labels ---")
    for count, num_videos in label_count_distribution.items():
        print(f"Videos with {int(count)} label(s): {num_videos}")
        
    multi_label_df = df[df['label_count'] > 1]
    
    if multi_label_df.empty:
        print("\nNo videos with multiple labels were found in this dataset.")
        return
        
    combinations = []
    for _, row in multi_label_df.iterrows():
        active_labels = [col for col in label_columns if row[col] == 1]
        sorted_labels = tuple(sorted(active_labels, key=lambda x: LABEL_ORDER.index(x)))
        combinations.append(sorted_labels)
        
    combination_counts = Counter(combinations)
    
    print("\n--- Top 5 Most Common Multi-Label Combinations ---")
    print(f"{'Label Codes':<20} | {'Combination':<45} | {'Number of Videos'}")
    print("-" * 85)
    
    for (combo_codes, count) in combination_counts.most_common(5):
        code_str = ' + '.join(combo_codes)
        name_str = ' + '.join([LABEL_MAP.get(code, "Unknown") for code in combo_codes])
        print(f"{code_str:<20} | {name_str:<45} | {count}")
    print("-" * 85)
    print("\n")


if __name__ == "__main__":
    analyze_multilabel_distribution(GROUND_TRUTH_CSV)

