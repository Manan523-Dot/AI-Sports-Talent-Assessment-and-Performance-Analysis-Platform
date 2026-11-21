
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

HEIGHT_COLUMNS = [
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right"
]


def main():
    groundtruth = pd.read_csv(GROUNDTRUTH_FILE, sep=";")

    records = []

    files = sorted((DATASET_DIR / "depthcamera").rglob("*.p"))

    for file_path in files:
        subject_id = int(file_path.parent.parent.name)
        jump_id = int(file_path.parent.name.replace("j", ""))

        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        pelvis_y = np.asarray(data["pelvis"]["y"], dtype=float)

        if len(pelvis_y) < 30 or not np.all(np.isfinite(pelvis_y)):
            continue

        start_baseline = float(np.median(pelvis_y[:15]))
        end_baseline = float(np.median(pelvis_y[-15:]))
        average_baseline = (start_baseline + end_baseline) / 2

        peak_y = float(np.min(pelvis_y))

        records.append({
            "id": subject_id,
            "jump": jump_id,
            "start_baseline_displacement": start_baseline - peak_y,
            "end_baseline_displacement": end_baseline - peak_y,
            "average_baseline_displacement": average_baseline - peak_y
        })

    features = pd.DataFrame(records)

    df = groundtruth.merge(features, on=["id", "jump"])

    print("ALL-RECORDING PELVIS BASELINE EVALUATION")
    print("=" * 65)
    print("Recordings processed:", len(df))

    methods = [
        "start_baseline_displacement",
        "end_baseline_displacement",
        "average_baseline_displacement"
    ]

    for method in methods:
        print(f"\n\nMETHOD: {method}")
        print("-" * 65)

        for target in HEIGHT_COLUMNS:
            valid = df[df[target] != -1]

            actual = valid[target].to_numpy(dtype=float)
            predicted = valid[method].to_numpy(dtype=float)

            mae = np.mean(np.abs(predicted - actual))
            bias = np.mean(predicted - actual)
            correlation = np.corrcoef(predicted, actual)[0, 1]

            print(f"\nTarget: {target}")
            print("Valid records:", len(valid))
            print("MAE (mm):", round(mae, 2))
            print("Bias (mm):", round(bias, 2))
            print("Correlation:", round(correlation, 3))

    print("\n\nEVALUATION COMPLETED")
    print("Note: This is a diagnostic comparison, not an ML test result.")


if __name__ == "__main__":
    main()
