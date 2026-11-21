
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_features.csv"
)

PREDICTION_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_regression_predictions.csv"
)

features = pd.read_csv(FEATURE_FILE)
predictions = pd.read_csv(PREDICTION_FILE)

df = predictions.merge(
    features[
    [
        "subject_id",
        "jump_id",
        "pelvis_baseline_displacement",
        "pelvis_vertical_range",
        "pelvis_vertical_std",
        "chest_vertical_range"
    ]
],
    on=["subject_id", "jump_id"],
    how="left"
)

participants = [22, 23, 1, 7]

for participant in participants:

    subset = df[df["subject_id"] == participant].copy()

    if subset.empty:
        continue

    print("\n" + "=" * 75)
    print("PARTICIPANT:", participant)
    print("=" * 75)

    columns = [
    "jump_id",
    "actual_height_mm",
    "pelvis_baseline_displacement",
    "pelvis_vertical_range",
    "pelvis_vertical_std",
    "chest_vertical_range",
    "linear_predicted_mm",
    "ridge_predicted_mm",
    "linear_absolute_error_mm",
    "ridge_absolute_error_mm"
]

    print(subset[columns].round(2).to_string(index=False))

print("\nINSPECTION COMPLETED")

