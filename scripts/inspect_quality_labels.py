
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "vertical_jump_features.csv"
)

df = pd.read_csv(INPUT_FILE)

print("QUALITY LABEL INSPECTION")
print("=" * 55)

# 1. Overall label distribution
print("\n1. OVERALL LABEL DISTRIBUTION")

print(df["well_executed"].value_counts().sort_index())

print("\nLabel percentages:")
print(
    (df["well_executed"].value_counts(normalize=True)
     .sort_index() * 100).round(2)
)

# 2. Label distribution by participant
print("\n2. LABEL DISTRIBUTION BY PARTICIPANT")

participant_summary = (
    df.groupby("subject_id")["well_executed"]
    .agg(
        total_recordings="count",
        positive_count="sum",
        unique_labels="nunique",
        positive_rate="mean"
    )
)

participant_summary["negative_count"] = (
    participant_summary["total_recordings"]
    - participant_summary["positive_count"]
)

all_positive = (
    participant_summary["positive_count"] == 5
).sum()

all_negative = (
    participant_summary["negative_count"] == 5
).sum()

mixed = (
    participant_summary["unique_labels"] > 1
).sum()

print("Participants with all positive labels:", all_positive)
print("Participants with all negative labels:", all_negative)
print("Participants with mixed labels:", mixed)

print("\nParticipant-wise details:")
print(participant_summary.to_string())

# 3. Label distribution by jump number
print("\n3. LABEL DISTRIBUTION BY JUMP NUMBER")

jump_summary = (
    df.groupby("jump_id")["well_executed"]
    .agg(
        recordings="count",
        positive_count="sum",
        positive_rate="mean"
    )
)

jump_summary["negative_count"] = (
    jump_summary["recordings"]
    - jump_summary["positive_count"]
)

jump_summary["positive_percentage"] = (
    jump_summary["positive_rate"] * 100
)

print(
    jump_summary[
        ["recordings", "positive_count",
         "negative_count", "positive_percentage"]
    ].round(2).to_string()
)

# 4. Save reports
output_file = (
    PROJECT_DIR / "data" / "processed"
    / "quality_label_inspection.csv"
)

participant_summary.to_csv(output_file)

print("\n" + "=" * 55)
print("INSPECTION COMPLETED")
print("Participant report saved to:")
print(output_file)
