from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "vertical_jump_features.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "feature_target_correlations.csv"
)

df = pd.read_csv(INPUT_FILE)

# Columns that must never be used as model inputs
excluded_columns = [
    "subject_id",
    "jump_id",
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right",
    "well_executed",
]

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
]

print("VERTICAL JUMP FEATURE ANALYSIS")
print("=" * 55)

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")
print("Total recordings:", len(df))
print("Unique participants:", df["subject_id"].nunique())
print("Extracted input features:", len(feature_columns))
print("Duplicate recording IDs:",
      df.duplicated(["subject_id", "jump_id"]).sum())

# 2. Recordings per participant
print("\n2. RECORDINGS PER PARTICIPANT")
recording_counts = df.groupby("subject_id").size()

print("Minimum:", recording_counts.min())
print("Maximum:", recording_counts.max())
print("Participants with five recordings:",
      (recording_counts == 5).sum())

# 3. Constant features
print("\n3. CONSTANT FEATURES")

constant_features = [
    col for col in feature_columns
    if df[col].nunique(dropna=True) <= 1
]

if constant_features:
    print(constant_features)
else:
    print("No constant features found.")

# 4. Height target statistics
print("\n4. JUMP HEIGHT STATISTICS")

height_targets = [
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right",
]

for target in height_targets:
    valid = df[df[target] != -1][target]

    print(f"\n{target}")
    print("Valid records:", len(valid))
    print("Minimum:", round(valid.min(), 2))
    print("Maximum:", round(valid.max(), 2))
    print("Mean:", round(valid.mean(), 2))
    print("Median:", round(valid.median(), 2))

# 5. Feature-target correlations
print("\n5. FEATURE-TARGET CORRELATIONS")

targets = height_targets + ["well_executed"]

correlation_results = []

for target in targets:
    valid_df = df[df[target] != -1] if target != "well_executed" else df

    correlations = (
        valid_df[feature_columns + [target]]
        .corr(numeric_only=True)[target]
        .drop(target)
    )

    correlations = correlations.dropna()

    print(f"\nTarget: {target}")
    print("Top 10 features by absolute correlation:")

    top_features = correlations.abs().sort_values(
        ascending=False
    ).head(10)

    for feature in top_features.index:
        value = correlations[feature]
        print(f"{feature}: {value:.4f}")

    for feature, value in correlations.items():
        correlation_results.append({
            "target": target,
            "feature": feature,
            "correlation": value,
        })

# Save correlation results
correlation_df = pd.DataFrame(correlation_results)
correlation_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 55)
print("ANALYSIS COMPLETED")
print("Correlation report saved to:")
print(OUTPUT_FILE)