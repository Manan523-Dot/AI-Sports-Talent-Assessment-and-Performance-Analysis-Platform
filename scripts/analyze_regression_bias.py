
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_regression_predictions.csv"
)

df = pd.read_csv(INPUT_FILE)

# Signed error: positive means overprediction,
# negative means underprediction.
df["linear_error"] = (
    df["linear_predicted_mm"] - df["actual_height_mm"]
)

df["ridge_error"] = (
    df["ridge_predicted_mm"] - df["actual_height_mm"]
)

# Divide actual jump heights into three ranges
df["height_group"] = pd.cut(
    df["actual_height_mm"],
    bins=[0, 300, 450, float("inf")],
    labels=["Low (below 300 mm)", "Medium (300-450 mm)", "High (above 450 mm)"]
)

print("REGRESSION BIAS ANALYSIS")
print("=" * 65)

summary = df.groupby("height_group", observed=True).agg(
    recordings=("actual_height_mm", "count"),
    actual_mean=("actual_height_mm", "mean"),
    linear_prediction_mean=("linear_predicted_mm", "mean"),
    ridge_prediction_mean=("ridge_predicted_mm", "mean"),
    linear_mae=("linear_absolute_error_mm", "mean"),
    ridge_mae=("ridge_absolute_error_mm", "mean"),
    linear_bias=("linear_error", "mean"),
    ridge_bias=("ridge_error", "mean")
)

print("\nERROR BY ACTUAL JUMP-HEIGHT GROUP")
print(summary.round(2).to_string())

print("\nOVERALL BIAS")
print("Linear bias:", round(df["linear_error"].mean(), 2), "mm")
print("Ridge bias:", round(df["ridge_error"].mean(), 2), "mm")

print("\nINTERPRETATION")
print("Positive bias = overprediction")
print("Negative bias = underprediction")

output_file = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_regression_bias_summary.csv"
)

summary.to_csv(output_file)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("Saved:", output_file)
