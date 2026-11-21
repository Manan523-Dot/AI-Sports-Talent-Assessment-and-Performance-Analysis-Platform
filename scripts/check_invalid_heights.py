
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR / "data" / "raw" /
    "TUW_2024-JumpMetric_-_Vertical_Jump_Height_Estimation"
)

df = pd.read_csv(DATASET_DIR / "groundtruth.csv", sep=";")

height_columns = [
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right"
]

print("INVALID HEIGHT ANALYSIS")
print("=" * 50)

for column in height_columns:
    invalid = (df[column] == -1).sum()
    print(f"{column}: {invalid} invalid records")

print("\nRows containing -1:")
invalid_rows = df[(df[height_columns] == -1).any(axis=1)]

print(invalid_rows.to_string(index=False))

print("\nValid height records:")
print(len(df) - len(invalid_rows))

print("\nWell-executed labels among invalid-height records:")
print(invalid_rows["well_executed"].value_counts())

print("\nWell-executed labels among valid-height records:")
valid_rows = df[~(df[height_columns] == -1).any(axis=1)]
print(valid_rows["well_executed"].value_counts())

print("\nANALYSIS COMPLETED")
