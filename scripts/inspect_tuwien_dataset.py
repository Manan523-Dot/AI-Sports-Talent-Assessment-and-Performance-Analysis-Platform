
from pathlib import Path
import csv

# Project dataset location
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "TUW_2024-JumpMetric_-_Vertical_Jump_Height_Estimation"
)


def inspect_csv(file_path, rows_to_show=6):
    print(f"\n--- {file_path.name} ---")

    with open(file_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file, delimiter=";")

        for index, row in enumerate(reader):
            print(row)

            if index >= rows_to_show:
                break


def main():
    print("TU WIEN DATASET INSPECTION")
    print("=" * 45)

    if not DATASET_DIR.exists():
        print("ERROR: Dataset folder not found!")
        print("Expected location:", DATASET_DIR)
        return

    print("\nDataset folder found:", DATASET_DIR)

    # Show main folders and files
    print("\n1. MAIN DATASET CONTENTS")

    for item in sorted(DATASET_DIR.iterdir()):
        print(item.name, "(folder)" if item.is_dir() else "(file)")

    # Count CSV files
    print("\n2. CSV FILE COUNTS")

    all_csv_files = list(DATASET_DIR.rglob("*.csv"))
    print("Total CSV files:", len(all_csv_files))

    # Inspect ground truth
    groundtruth = DATASET_DIR / "groundtruth.csv"

    if groundtruth.exists():
        print("\n3. GROUND TRUTH DATA")
        inspect_csv(groundtruth, rows_to_show=10)
    else:
        print("\nWARNING: groundtruth.csv not found")

    # Inspect subject information
    subjects = DATASET_DIR / "subjects.csv"

    if subjects.exists():
        print("\n4. SUBJECT INFORMATION")
        inspect_csv(subjects, rows_to_show=5)
    else:
        print("\nWARNING: subjects.csv not found")

    # Inspect one depth-camera sample
    depth_folder = DATASET_DIR / "depthcamera"

    if depth_folder.exists():
        sample_files = list(depth_folder.rglob("s01_j1_hip_left.csv"))

        if sample_files:
            print("\n5. DEPTH CAMERA SAMPLE")
            inspect_csv(sample_files[0], rows_to_show=5)
        else:
            print("\nWARNING: Sample hip CSV not found")

    # Check timestamp-related filenames
    print("\n6. POSSIBLE TIMESTAMP FILES")

    timestamp_files = [
        file.name
        for file in all_csv_files
        if "time" in file.name.lower()
    ]

    if timestamp_files:
        for name in timestamp_files[:15]:
            print(name)
    else:
        print("No CSV filenames containing 'time' were found.")

    print("\nINSPECTION COMPLETED")


if __name__ == "__main__":
    main()
