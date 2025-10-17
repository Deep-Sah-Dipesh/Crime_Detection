import pandas as pd
import os

GROUND_TRUTH_CSV = r"H:\Crime_Detection\data\testing_ground_truth.csv"

LABEL_MAP = {
    'A': 'Normal', 'B1': 'Fighting', 'B2': 'Shooting', 'B4': 'Riot',
    'B5': 'Abuse', 'B6': 'Car Accident', 'G': 'Explosion'
}

DESCRIPTION_MAP = {
    'A': 'General, non-violent, or everyday scenes.',
    'B1': 'Physical altercations, hand-to-hand combat, brawls.',
    'B2': 'Scenes involving the use of firearms or gunfights.',
    'B4': 'Large-scale public disturbances, civil unrest.',
    'B5': 'Scenes depicting verbal or physical mistreatment.',
    'B6': 'Vehicle collisions, crashes, and related incidents.',
    'G': 'Detonations, blasts, and explosions.'
}

# Define the exact order for the output table.
LABEL_ORDER = ['A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G']

def analyze_label_distribution(csv_path):
    """
    Reads the ground truth CSV and calculates the number of videos per label.
    """
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    print("--- Analyzing Single-Label Distribution ---")
    print(f"Source CSV: {csv_path}")
    print(f"Total Unique Videos Analyzed: {len(df)}")
    
    label_counts = {}
    
    label_columns = [col for col in df.columns if col in LABEL_MAP]

    for col_code in label_columns:
        count = df[col_code].sum()
        label_counts[col_code] = int(count)

    print(f"\n{'Label Name':<15} | {'Category':<10} | {'Number of Videos':<20} | {'Description'}")
    print("-" * 100)
    for code in LABEL_ORDER:
        if code in label_counts:
            count = label_counts[code]
            label_name = LABEL_MAP.get(code, "Unknown")
            description = DESCRIPTION_MAP.get(code, "No description available.")
            print(f"{label_name:<15} | {code:<10} | {count:<20} | {description}")
    print("-" * 100)
    print("\nNote: The sum of counts may be greater than the total number of unique videos")
    print("because this is a multi-label dataset.\n")


if __name__ == "__main__":
    analyze_label_distribution(GROUND_TRUTH_CSV)

