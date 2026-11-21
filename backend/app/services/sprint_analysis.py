import json
import os

import numpy as np
import pandas as pd


ANGLE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_trunk_angle",
    "right_trunk_angle",
]

MIN_VALID_FRAMES = 5


def safe_number(value):
    try:
        number = float(value)
        if np.isfinite(number):
            return round(number, 2)
    except (TypeError, ValueError):
        pass
    return None


def calculate_side_rom(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    # Ignore measurements outside the possible angle range.
    values = values[(values >= 0) & (values <= 180)]

    if len(values) < MIN_VALID_FRAMES:
        return None, len(values)

    return safe_number(values.max() - values.min()), len(values)


def calculate_paired_symmetry(left, right):
    paired = pd.concat(
        [
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
        ],
        axis=1,
    ).dropna()

    paired.columns = ["left", "right"]

    paired = paired[
        paired["left"].between(0, 180)
        & paired["right"].between(0, 180)
    ]

    if len(paired) < MIN_VALID_FRAMES:
        return None, len(paired)

    left_rom = paired["left"].max() - paired["left"].min()
    right_rom = paired["right"].max() - paired["right"].min()

    return safe_number(abs(left_rom - right_rom)), len(paired)


def calculate_movement(df, left_column, right_column):
    left_rom, left_count = calculate_side_rom(df[left_column])
    right_rom, right_count = calculate_side_rom(df[right_column])

    available_roms = [
        value for value in [left_rom, right_rom]
        if value is not None
    ]

    average_rom = (
        safe_number(np.mean(available_roms))
        if available_roms else None
    )

    symmetry, paired_count = calculate_paired_symmetry(
        df[left_column],
        df[right_column],
    )

    return {
        "left_rom": left_rom,
        "right_rom": right_rom,
        "average_rom": average_rom,
        "symmetry_difference": symmetry,
        "left_valid_frames": left_count,
        "right_valid_frames": right_count,
        "paired_valid_frames": paired_count,
        "available": average_rom is not None,
        "symmetry_available": symmetry is not None,
    }


def analyze_sprint(input_csv, output_json):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Sprint CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if "frame" not in df.columns or "time_seconds" not in df.columns:
        raise ValueError(
            "Sprint CSV must contain frame and time_seconds columns."
        )

    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["time_seconds"] = pd.to_numeric(
        df["time_seconds"], errors="coerce"
    )

    df = df.dropna(subset=["frame", "time_seconds"]).copy()
    df = df.sort_values("time_seconds").reset_index(drop=True)

    if len(df) < MIN_VALID_FRAMES:
        raise ValueError("Not enough valid frames to analyze this sprint video.")

    # Missing angle columns are allowed.
    for column in ANGLE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

        df[column] = pd.to_numeric(df[column], errors="coerce")

        # Invalid angles become unavailable measurements.
        df.loc[
            ~df[column].between(0, 180),
            column
        ] = np.nan

        # Smooth available values without inventing missing measurements.
        df[column] = df[column].rolling(
            window=5,
            center=True,
            min_periods=1,
        ).median()

    duration = safe_number(
        df["time_seconds"].max() - df["time_seconds"].min()
    )

    knee = calculate_movement(
        df, "left_knee_angle", "right_knee_angle"
    )

    hip = calculate_movement(
        df, "left_hip_angle", "right_hip_angle"
    )

    trunk = calculate_movement(
        df, "left_trunk_angle", "right_trunk_angle"
    )

    # This is descriptive angle variability, not stride consistency.
    available_knee_values = pd.concat(
        [
            df["left_knee_angle"],
            df["right_knee_angle"],
        ],
        ignore_index=True,
    ).dropna()

    knee_angle_variability = None

    if len(available_knee_values) >= MIN_VALID_FRAMES:
        knee_angle_variability = safe_number(
            available_knee_values.std()
        )

    visibility_counts = {
        column: int(df[column].notna().sum())
        for column in ANGLE_COLUMNS
    }

    detected_sides = {
        "left": any(
            visibility_counts[column] >= MIN_VALID_FRAMES
            for column in [
                "left_knee_angle",
                "left_hip_angle",
                "left_trunk_angle",
            ]
        ),
        "right": any(
            visibility_counts[column] >= MIN_VALID_FRAMES
            for column in [
                "right_knee_angle",
                "right_hip_angle",
                "right_trunk_angle",
            ]
        ),
    }

    feature_availability = {
        "knee_movement": knee["available"],
        "knee_symmetry": knee["symmetry_available"],
        "hip_movement": hip["available"],
        "hip_symmetry": hip["symmetry_available"],
        "trunk_movement": trunk["available"],
        "knee_angle_variability": knee_angle_variability is not None,
    }

    # Keep the existing chart key and include the average for the frontend.
    knee_angle_data = []

    for index, row in df.iterrows():
        left = safe_number(row["left_knee_angle"])
        right = safe_number(row["right_knee_angle"])

        available_angles = [
            value for value in [left, right]
            if value is not None
        ]

        average = (
            safe_number(np.mean(available_angles))
            if available_angles else None
        )

        knee_angle_data.append({
            "frame": int(row["frame"]),
            "time_seconds": safe_number(row["time_seconds"]),
            "left_knee_angle": left,
            "right_knee_angle": right,
            "average_knee_angle": average,
        })

    result = {
        "activity": "Sprint",
        "frame_count": int(len(df)),
        "duration_seconds": duration,

        "knee_movement": knee,
        "hip_movement": hip,
        "trunk_movement": trunk,

        # Kept for compatibility with existing consumers.
        "movement_consistency": knee_angle_variability,
        "knee_angle_variability": knee_angle_variability,
        "movement_consistency_label": (
            "Descriptive knee-angle variability; "
            "not validated stride-to-stride consistency"
        ),

        "visibility_counts": visibility_counts,
        "detected_sides": detected_sides,
        "feature_availability": feature_availability,

        "knee_angle_data": knee_angle_data,
    }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)

    with open(output_json, "w") as file:
        json.dump(result, file, indent=4, allow_nan=False)

    print("\nSPRINT ANALYSIS COMPLETE")
    print("Frames:", len(df))
    print("Duration:", duration)
    print("Knee ROM:", knee["average_rom"])
    print("Hip ROM:", hip["average_rom"])
    print("Detected sides:", detected_sides)

    return result


if __name__ == "__main__":
    analyze_sprint(
        "data/processed/sprint_angles.csv",
        "data/processed/sprint_features.json",
    )