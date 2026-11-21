
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR / "data" / "raw" /
    "TUW_2024-JumpMetric_-_Vertical_Jump_Height_Estimation"
)

GROUNDTRUTH_FILE = DATASET_DIR / "groundtruth.csv"


def main():
    groundtruth = pd.read_csv(GROUNDTRUTH_FILE, sep=";")

    print("PELVIS DISPLACEMENT VS GROUND TRUTH")
    print("=" * 75)

    for jump in range(1, 6):
        file_path = (
            DATASET_DIR / "depthcamera" / "01" /
            f"j{jump}" / f"s01_j{jump}.p"
        )

        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        pelvis_y = np.asarray(data["pelvis"]["y"], dtype=float)

        # Estimate standing baseline from the beginning and end.
        start_baseline = np.median(pelvis_y[:15])
        end_baseline = np.median(pelvis_y[-15:])
        baseline = (start_baseline + end_baseline) / 2

        # Smaller Y means a higher position.
        peak_index = int(np.argmin(pelvis_y))
        peak_y = pelvis_y[peak_index]

        estimated_displacement = baseline - peak_y

        row = groundtruth[
            (groundtruth["id"] == 1) &
            (groundtruth["jump"] == jump)
        ].iloc[0]

        measured = row["height_measured"]
        corrected = row["height_cor_frame_center"]

        print(f"\nJump {jump}")
        print("-" * 50)
        print("Frames:", len(pelvis_y))
        print("Starting baseline:", round(start_baseline, 2))
        print("Ending baseline:", round(end_baseline, 2))
        print("Average baseline:", round(baseline, 2))
        print("Peak Y:", round(peak_y, 2))
        print("Peak frame:", peak_index)
        print("Pelvis displacement:", round(estimated_displacement, 2))
        print("Measured ground truth:", measured)
        print("Corrected centre ground truth:", corrected)

    print("\nCOMPARISON COMPLETED")


if __name__ == "__main__":
    main()
