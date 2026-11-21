
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_regression_predictions.csv"
)

df = pd.read_csv(INPUT_FILE)

print("PARTICIPANT-WISE REGRESSION ERROR ANALYSIS")
print("=" * 65)

# Calculate signed errors
df["linear_error_mm"] = (
    df["linear_predicted_mm"] - df["actual_height_mm"]
)

df["ridge_error_mm"] = (
    df["ridge_predicted_mm"] - df["actual_height_mm"]
)

# Summarize errors for each participant
summary = df.groupby("subject_id").agg(
    recordings=("jump_id", "count"),
    actual_mean=("actual_height_mm", "mean"),
    linear_mae=("linear_absolute_error_mm", "mean"),
    ridge_mae=("ridge_absolute_error_mm", "mean"),
    linear_bias=("linear_error_mm", "mean"),
    ridge_bias=("ridge_error_mm", "mean")
)

summary = summary.sort_values("ridge_mae", ascending=False)

print("\nTOP 10 PARTICIPANTS WITH HIGHEST RIDGE MAE")
print(summary.head(10).round(2).to_string())

print("\nOVERALL PARTICIPANT-WISE SUMMARY")
print("Average participant Ridge MAE:",
      round(summary["ridge_mae"].mean(), 2), "mm")

print("Average participant Linear MAE:",
      round(summary["linear_mae"].mean(), 2), "mm")

print("\nBIAS INTERPRETATION")
print("Positive bias = model tends to overpredict.")
print("Negative bias = model tends to underpredict.")

output_file = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_participant_error_summary.csv"
)

summary.to_csv(output_file)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("Saved:", output_file)
