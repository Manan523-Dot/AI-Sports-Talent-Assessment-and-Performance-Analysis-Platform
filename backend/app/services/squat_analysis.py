
import json
from pathlib import Path

import pandas as pd


# =============================================================
# CONFIGURATION
# =============================================================

STANDING_THRESHOLD = 160
DEEP_SQUAT_THRESHOLD = 90
CONFIRMATION_FRAMES = 2
SMOOTHING_WINDOW = 5
MIN_SYMMETRY_FRAMES = 5


# =============================================================
# HELPERS
# =============================================================

def safe_round(value, digits=2):
    if value is None or pd.isna(value):
        return None

    return round(float(value), digits)


def calculate_rom(series):
    values = series.dropna()

    if len(values) < 2:
        return None

    return float(values.max() - values.min())


def calculate_paired_symmetry(df, left_column, right_column):

    paired = df.dropna(
        subset=[left_column, right_column]
    )

    frame_count = len(paired)

    if frame_count < MIN_SYMMETRY_FRAMES:
        return {
            "left_rom": None,
            "right_rom": None,
            "difference": None,
            "frame_count": frame_count,
            "available": False
        }

    left_rom = calculate_rom(paired[left_column])
    right_rom = calculate_rom(paired[right_column])

    if left_rom is None or right_rom is None:
        return {
            "left_rom": None,
            "right_rom": None,
            "difference": None,
            "frame_count": frame_count,
            "available": False
        }

    return {
        "left_rom": left_rom,
        "right_rom": right_rom,
        "difference": abs(left_rom - right_rom),
        "frame_count": frame_count,
        "available": True
    }


# =============================================================
# REPETITION DETECTION
# =============================================================

def detect_squat_repetitions(df):

    repetitions = []

    state = "STANDING"

    start_index = None
    bottom_index = None

    deep_confirmation = 0
    standing_confirmation = 0

    knee_angles = df["smoothed_knee_angle"].tolist()

    for index, angle in enumerate(knee_angles):

        if pd.isna(angle):
            continue

        if state == "STANDING":

            if angle < STANDING_THRESHOLD:

                start_index = index
                bottom_index = index

                deep_confirmation = 0
                standing_confirmation = 0

                state = "DESCENDING"

        elif state == "DESCENDING":

            if angle < knee_angles[bottom_index]:
                bottom_index = index

            if angle <= DEEP_SQUAT_THRESHOLD:
                deep_confirmation += 1
            else:
                deep_confirmation = 0

            if deep_confirmation >= CONFIRMATION_FRAMES:
                state = "BOTTOM"

        elif state == "BOTTOM":

            if angle < knee_angles[bottom_index]:
                bottom_index = index

            if angle > DEEP_SQUAT_THRESHOLD:
                state = "ASCENDING"

        elif state == "ASCENDING":

            if angle < knee_angles[bottom_index]:
                bottom_index = index

            if angle <= DEEP_SQUAT_THRESHOLD:

                state = "BOTTOM"
                standing_confirmation = 0

            elif angle >= STANDING_THRESHOLD:

                standing_confirmation += 1

                if standing_confirmation >= CONFIRMATION_FRAMES:

                    end_index = index

                    start_time = float(
                        df.iloc[start_index]["time_seconds"]
                    )

                    bottom_time = float(
                        df.iloc[bottom_index]["time_seconds"]
                    )

                    end_time = float(
                        df.iloc[end_index]["time_seconds"]
                    )

                    descent_time = bottom_time - start_time
                    ascent_time = end_time - bottom_time
                    rep_time = end_time - start_time

                    if (
                        descent_time > 0
                        and ascent_time > 0
                        and rep_time > 0
                    ):

                        repetitions.append({
                            "rep_number": len(repetitions) + 1,
                            "start_time": safe_round(start_time),
                            "bottom_time": safe_round(bottom_time),
                            "end_time": safe_round(end_time),
                            "descent_time": safe_round(descent_time),
                            "ascent_time": safe_round(ascent_time),
                            "rep_time": safe_round(rep_time),
                            "bottom_angle": safe_round(
                                knee_angles[bottom_index]
                            )
                        })

                    state = "STANDING"

                    start_index = None
                    bottom_index = None

                    deep_confirmation = 0
                    standing_confirmation = 0

            else:
                standing_confirmation = 0

    return repetitions


# =============================================================
# MAIN SQUAT ANALYSIS
# =============================================================

def analyze_squat(input_csv, output_json):

    input_csv = Path(input_csv)
    output_json = Path(output_json)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Squat angle CSV not found: {input_csv}"
        )

    df = pd.read_csv(input_csv)

    required_columns = [
        "time_seconds",
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_trunk_angle",
        "right_trunk_angle"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # CONVERT TO NUMERIC
    # ---------------------------------------------------------

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.sort_values(
        "time_seconds"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # MEASUREMENT VISIBILITY
    # ---------------------------------------------------------

    visibility_counts = {
        column: int(df[column].notna().sum())
        for column in [
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
            "left_trunk_angle",
            "right_trunk_angle"
        ]
    }

    left_detected = (
        visibility_counts["left_knee_angle"] > 0
        and visibility_counts["left_hip_angle"] > 0
    )

    right_detected = (
        visibility_counts["right_knee_angle"] > 0
        and visibility_counts["right_hip_angle"] > 0
    )

    if left_detected and right_detected:
        detected_sides = "both"
    elif left_detected:
        detected_sides = "left"
    elif right_detected:
        detected_sides = "right"
    else:
        detected_sides = "insufficient"

    # ---------------------------------------------------------
    # ADAPTIVE JOINT ANGLES
    #
    # Use both sides when available.
    # If only one side is visible, use that side.
    # ---------------------------------------------------------

    df["average_knee_angle"] = df[
        ["left_knee_angle", "right_knee_angle"]
    ].mean(axis=1)

    df["average_hip_angle"] = df[
        ["left_hip_angle", "right_hip_angle"]
    ].mean(axis=1)

    df["average_trunk_angle"] = df[
        ["left_trunk_angle", "right_trunk_angle"]
    ].mean(axis=1)

    df = df.dropna(
        subset=["time_seconds", "average_knee_angle"]
    ).reset_index(drop=True)

    if len(df) < 5:
        raise ValueError(
            "Not enough valid knee angle data for squat analysis."
        )

    # ---------------------------------------------------------
    # INTERPOLATE HIP AND TRUNK DATA
    # ---------------------------------------------------------

    for column in [
        "average_hip_angle",
        "average_trunk_angle"
    ]:

        if df[column].notna().any():
            df[column] = df[column].interpolate(
                limit_direction="both"
            )

    # ---------------------------------------------------------
    # SMOOTH ANGLES
    # ---------------------------------------------------------

    df["smoothed_knee_angle"] = (
        df["average_knee_angle"]
        .rolling(
            window=SMOOTHING_WINDOW,
            center=True,
            min_periods=1
        )
        .mean()
    )

    df["smoothed_hip_angle"] = (
        df["average_hip_angle"]
        .rolling(
            window=SMOOTHING_WINDOW,
            center=True,
            min_periods=1
        )
        .mean()
    )

    df["smoothed_trunk_angle"] = (
        df["average_trunk_angle"]
        .rolling(
            window=SMOOTHING_WINDOW,
            center=True,
            min_periods=1
        )
        .mean()
    )

    # =========================================================
    # 1. REPETITION ANALYSIS
    # =========================================================

    rep_details = detect_squat_repetitions(df)

    repetitions = len(rep_details)

    rep_times = [
        rep["rep_time"] for rep in rep_details
    ]

    descent_times = [
        rep["descent_time"] for rep in rep_details
    ]

    ascent_times = [
        rep["ascent_time"] for rep in rep_details
    ]

    bottom_angles = [
        rep["bottom_angle"] for rep in rep_details
    ]

    if repetitions > 0:

        average_rep_time = sum(rep_times) / repetitions
        fastest_rep_time = min(rep_times)
        slowest_rep_time = max(rep_times)

        average_descent_time = sum(descent_times) / repetitions
        average_ascent_time = sum(ascent_times) / repetitions

        average_rep_bottom_angle = (
            sum(bottom_angles) / repetitions
        )

        deepest_rep_angle = min(bottom_angles)

        rep_time_difference = (
            slowest_rep_time - fastest_rep_time
        )

        deepest_rep = min(
            rep_details,
            key=lambda rep: rep["bottom_angle"]
        )

        deepest_point_time = deepest_rep["bottom_time"]

    else:

        average_rep_time = None
        fastest_rep_time = None
        slowest_rep_time = None

        average_descent_time = None
        average_ascent_time = None

        average_rep_bottom_angle = None
        deepest_rep_angle = None

        rep_time_difference = None
        deepest_point_time = None

    # =========================================================
    # 2. KNEE RANGE OF MOTION
    # =========================================================

    knee_rom = calculate_rom(
        df["smoothed_knee_angle"]
    )

    # =========================================================
    # 3. KNEE SYMMETRY
    # =========================================================

    knee_symmetry = calculate_paired_symmetry(
        df,
        "left_knee_angle",
        "right_knee_angle"
    )

    # =========================================================
    # 4. HIP RANGE OF MOTION
    # =========================================================

    hip_rom = calculate_rom(
        df["smoothed_hip_angle"]
    )

    # =========================================================
    # 5. HIP SYMMETRY
    # =========================================================

    hip_symmetry = calculate_paired_symmetry(
        df,
        "left_hip_angle",
        "right_hip_angle"
    )

    # =========================================================
    # 6. TRUNK MOVEMENT
    # =========================================================

    trunk_values = df[
        "smoothed_trunk_angle"
    ].dropna()

    if len(trunk_values) >= 2:

        trunk_angle_change = (
            trunk_values.max() - trunk_values.min()
        )

        average_trunk_angle = trunk_values.mean()

    else:

        trunk_angle_change = None
        average_trunk_angle = None

    # =========================================================
    # 7. PREPARE FEATURES
    # =========================================================

    features = {

        "repetitions": repetitions,

        "detected_sides": detected_sides,

        "visibility_counts": visibility_counts,

        "knee_rom": safe_round(knee_rom),
        "hip_rom": safe_round(hip_rom),

        "left_knee_rom": safe_round(
            knee_symmetry["left_rom"]
        ),

        "right_knee_rom": safe_round(
            knee_symmetry["right_rom"]
        ),

        "knee_rom_difference": safe_round(
            knee_symmetry["difference"]
        ),

        "knee_symmetry_frames": knee_symmetry[
            "frame_count"
        ],

        "knee_symmetry_available": knee_symmetry[
            "available"
        ],

        "left_hip_rom": safe_round(
            hip_symmetry["left_rom"]
        ),

        "right_hip_rom": safe_round(
            hip_symmetry["right_rom"]
        ),

        "hip_rom_difference": safe_round(
            hip_symmetry["difference"]
        ),

        "hip_symmetry_frames": hip_symmetry[
            "frame_count"
        ],

        "hip_symmetry_available": hip_symmetry[
            "available"
        ],

        "average_rep_bottom_angle": safe_round(
            average_rep_bottom_angle
        ),

        "deepest_rep_angle": safe_round(
            deepest_rep_angle
        ),

        "deepest_point_time": safe_round(
            deepest_point_time
        ),

        "average_descent_time": safe_round(
            average_descent_time
        ),

        "average_ascent_time": safe_round(
            average_ascent_time
        ),

        "descent_time": safe_round(
            average_descent_time
        ),

        "ascent_time": safe_round(
            average_ascent_time
        ),

        "average_rep_time": safe_round(
            average_rep_time
        ),

        "fastest_rep_time": safe_round(
            fastest_rep_time
        ),

        "slowest_rep_time": safe_round(
            slowest_rep_time
        ),

        "rep_time_difference": safe_round(
            rep_time_difference
        ),

        "trunk_angle_change": safe_round(
            trunk_angle_change
        ),

        "average_trunk_angle": safe_round(
            average_trunk_angle
        ),

        "rep_details": rep_details
    }

    # =========================================================
    # 8. SAVE FEATURES
    # =========================================================

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as json_file:

        json.dump(
            features,
            json_file,
            indent=4
        )

    print("\nSQUAT ANALYSIS COMPLETED")
    print("=" * 55)

    print("Repetitions:", repetitions)
    print("Detected sides:", detected_sides)
    print("Knee ROM:", features["knee_rom"])
    print("Hip ROM:", features["hip_rom"])

    print(
        "Knee symmetry difference:",
        features["knee_rom_difference"]
    )

    print(
        "Hip symmetry difference:",
        features["hip_rom_difference"]
    )

    print(
        "Knee paired frames:",
        features["knee_symmetry_frames"]
    )

    print(
        "Hip paired frames:",
        features["hip_symmetry_frames"]
    )

    print("Output:", output_json)

    return features


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parents[3]

    input_file = (
        project_dir
        / "data"
        / "processed"
        / "squat_angles.csv"
    )

    output_file = (
        project_dir
        / "data"
        / "processed"
        / "squat_features.json"
    )

    analyze_squat(
        input_file,
        output_file
    )
