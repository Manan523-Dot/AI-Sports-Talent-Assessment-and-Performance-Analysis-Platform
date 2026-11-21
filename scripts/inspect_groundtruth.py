
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR / "data" / "raw" /
    "TUW_2024-JumpMetric_-_Vertical_Jump_Height_Estimation"
)

groundtruth_path = DATASET_DIR / "groundtruth.csv"

df = pd.read_csv(groundtruth_path, sep=";")

print("GROUND-TRUTH INSPECTION")
print("=" * 50)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

print("\nMissing values:")
print(df.isnull().sum())

print("\nJump-height statistics (all height columns):")
height_columns = [
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right"
]

print(df[height_columns].describe().round(2))

print("\nWell-executed label distribution:")
print(df["well_executed"].value_counts(dropna=False))

print("\nUnique label values:")
print(df["well_executed"].unique())

print("\nGround-truth inspection completed.")
