
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "vertical_jump_candidate_oof_predictions.csv"
)

df = pd.read_csv(INPUT_FILE)

df["signed_error_mm"] = (
    df["predicted_height_mm"] - df["actual_height_mm"]
)

df["absolute_error_mm"] = df["signed_error_mm"].abs()

print("OUT-OF-FOLD PREDICTION ERROR ANALYSIS")
print("=" * 75)

# 1. Overall model comparison
print("\n1. OVERALL MODEL PERFORMANCE")

overall = df.groupby("model").agg(
    recordings=("actual_height_mm", "count"),
    MAE=("absolute_error_mm", "mean"),
    RMSE=("signed_error_mm", lambda x: (x.pow(2).mean()) ** 0.5),
    bias=("signed_error_mm", "mean"),
)

print(overall.round(2).to_string())

# 2. Participant-wise errors
print("\n2. PARTICIPANT-WISE MAE")

participant_results = df.groupby(
    ["model", "subject_id"]
).agg(
    recordings=("actual_height_mm", "count"),
    actual_mean=("actual_height_mm", "mean"),
    MAE=("absolute_error_mm", "mean"),
    bias=("signed_error_mm", "mean"),
).reset_index()

for model_name in df["model"].unique():

    print(f"\nMODEL: {model_name}")

    result = participant_results[
        participant_results["model"] == model_name
    ]

    print(
        result.sort_values(
            "MAE", ascending=False
        ).head(10).round(2).to_string(index=False)
    )

# 3. Height-range analysis
print("\n3. ERROR BY ACTUAL JUMP-HEIGHT RANGE")

def height_group(height):
    if height < 300:
        return "Low (<300 mm)"
    elif height <= 450:
        return "Medium (300-450 mm)"
    return "High (>450 mm)"

df["height_group"] = df["actual_height_mm"].apply(height_group)

height_results = df.groupby(
    ["model", "height_group"]
).agg(
    recordings=("actual_height_mm", "count"),
    actual_mean=("actual_height_mm", "mean"),
    predicted_mean=("predicted_height_mm", "mean"),
    MAE=("absolute_error_mm", "mean"),
    bias=("signed_error_mm", "mean"),
).reset_index()

print(height_results.round(2).to_string(index=False))

# 4. Save detailed results
output_file = (
    PROJECT_DIR / "data" / "processed"
    / "vertical_jump_candidate_error_analysis.csv"
)

participant_results.to_csv(output_file, index=False)

print("\n" + "=" * 75)
print("ANALYSIS COMPLETED")
print("Saved:", output_file)
