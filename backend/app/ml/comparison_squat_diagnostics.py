
import json
from pathlib import Path

import pandas as pd


STANDING_THRESHOLD = 160
BOTTOM_THRESHOLD = 90
CONFIRMATION_FRAMES = 2
SMOOTHING_WINDOW = 5


def diagnose_squat_csv(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    required = [
        "time_seconds",
        "left_knee_angle",
        "right_knee_angle"
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df = df.copy()

    df["left_knee_angle"] = pd.to_numeric(
        df["left_knee_angle"], errors="coerce"
    )
    df["right_knee_angle"] = pd.to_numeric(
        df["right_knee_angle"], errors="coerce"
    )

    df["knee_angle"] = df[
        ["left_knee_angle", "right_knee_angle"]
    ].mean(axis=1)

    df = df.dropna(subset=["time_seconds", "knee_angle"])
    df = df.sort_values("time_seconds").reset_index(drop=True)

    df["smoothed_angle"] = df["knee_angle"].rolling(
        SMOOTHING_WINDOW,
        center=True,
        min_periods=1
    ).mean()

    angles = df["smoothed_angle"].tolist()
    times = df["time_seconds"].tolist()

    state = "STANDING"
    bottom_frames = 0
    standing_frames = 0

    repetitions = 0
    reached_bottom = False
    started_descending = False
    returned_to_standing = False

    events = []

    for index, angle in enumerate(angles):

        timestamp = round(float(times[index]), 3)

        if state == "STANDING":

            if angle < STANDING_THRESHOLD:
                state = "DESCENDING"
                started_descending = True

                events.append({
                    "event": "Descent started",
                    "time": timestamp,
                    "angle": round(float(angle), 2)
                })

        elif state == "DESCENDING":

            if angle <= BOTTOM_THRESHOLD:
                bottom_frames += 1
            else:
                bottom_frames = 0

            if bottom_frames >= CONFIRMATION_FRAMES:
                state = "BOTTOM"
                reached_bottom = True

                events.append({
                    "event": "Bottom reached",
                    "time": timestamp,
                    "angle": round(float(angle), 2)
                })

        elif state == "BOTTOM":

            if angle > BOTTOM_THRESHOLD:
                state = "ASCENDING"
                standing_frames = 0

                events.append({
                    "event": "Ascent started",
                    "time": timestamp,
                    "angle": round(float(angle), 2)
                })

        elif state == "ASCENDING":

            if angle <= BOTTOM_THRESHOLD:
                state = "BOTTOM"
                standing_frames = 0

            elif angle >= STANDING_THRESHOLD:
                standing_frames += 1
            else:
                standing_frames = 0

            if standing_frames >= CONFIRMATION_FRAMES:
                repetitions += 1
                returned_to_standing = True
                state = "STANDING"

                events.append({
                    "event": "Complete repetition",
                    "time": timestamp,
                    "angle": round(float(angle), 2)
                })

                bottom_frames = 0
                standing_frames = 0

    if repetitions > 0:
        status = "Complete movement detected"
        reason = "At least one complete squat cycle was detected."

    elif not started_descending:
        status = "No squat movement detected"
        reason = "The video did not show a clear descent."

    elif not reached_bottom:
        status = "Bottom position not reached"
        reason = (
            "The knee angle did not remain at or below "
            "90 degrees for the required frames."
        )

    elif not returned_to_standing:
        status = "Incomplete squat"
        reason = (
            "The bottom position was reached, but the video "
            "did not show a complete return to standing."
        )

    else:
        status = "Incomplete movement"
        reason = "A complete squat cycle could not be confirmed."

    return {
        "filename": csv_path.name,
        "total_frames": len(df),
        "duration_seconds": round(
            float(times[-1] - times[0]), 3
        ) if len(times) > 1 else 0,
        "minimum_knee_angle": round(
            float(df["smoothed_angle"].min()), 2
        ),
        "maximum_knee_angle": round(
            float(df["smoothed_angle"].max()), 2
        ),
        "detected_repetitions": repetitions,
        "final_state": state,
        "status": status,
        "reason": reason,
        "events": events
    }


if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parents[3]
    processed_dir = project_dir / "data" / "processed"

    csv_files = [
        processed_dir / "3613d225-d282-49e6-a40f-d45d9d17687d_angles.csv",
        processed_dir / "5ae02db3-301e-42b3-a4fb-dcd0688343e4_angles.csv"
    ]

    for csv_file in csv_files:

        if not csv_file.exists():
            print(f"\nFile not found: {csv_file}")
            continue

        result = diagnose_squat_csv(csv_file)

        print("\n" + "=" * 55)
        print(json.dumps(result, indent=4))
