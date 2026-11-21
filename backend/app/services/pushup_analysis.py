import json
from pathlib import Path

import pandas as pd


TOP_THRESHOLD = 155
BOTTOM_THRESHOLD = 95
ASCENDING_THRESHOLD = 105

SMOOTHING_WINDOW = 5
CONFIRMATION_FRAMES = 2
MIN_VALID_FRAMES = 5
MIN_MEASUREMENT_COVERAGE = 0.20


COMPONENT_WEIGHTS = {
    "depth_score": 0.30,
    "alignment_score": 0.25,
    "symmetry_score": 0.20,
    "tempo_score": 0.15,
    "consistency_score": 0.10,
}


def safe_float(value):
    try:
        result = float(value)
        if pd.notna(result) and abs(result) != float("inf"):
            return result
    except (TypeError, ValueError):
        pass
    return None


def safe_round(value, digits=2):
    value = safe_float(value)
    return round(value, digits) if value is not None else None


def calculate_repetitions(df, angle_column):
    reps = []

    state = "top"
    candidate_count = 0

    start_time = None
    candidate_start_time = None
    bottom_time = None
    bottom_angle = None

    rep_number = 0

    for _, row in df.iterrows():

        angle = row[angle_column]
        timestamp = row["time_seconds"]

        if pd.isna(angle) or pd.isna(timestamp):
            continue

        # TOP: confirm the beginning of descent
        if state == "top":

            if angle < TOP_THRESHOLD:

                candidate_count += 1

                if candidate_count == 1:
                    candidate_start_time = timestamp

                if candidate_count >= CONFIRMATION_FRAMES:

                    state = "descending"
                    start_time = candidate_start_time
                    bottom_angle = angle
                    bottom_time = timestamp

            else:
                candidate_count = 0
                candidate_start_time = None

        # DESCENDING: track minimum elbow angle
        elif state == "descending":

            if angle < bottom_angle:
                bottom_angle = angle
                bottom_time = timestamp

            if angle <= BOTTOM_THRESHOLD:
                state = "bottom"

            elif angle >= TOP_THRESHOLD:

                # Incomplete repetition
                state = "top"
                candidate_count = 0
                start_time = None
                candidate_start_time = None
                bottom_angle = None
                bottom_time = None

        # BOTTOM: detect upward movement
        elif state == "bottom":

            if angle < bottom_angle:
                bottom_angle = angle
                bottom_time = timestamp

            if angle >= ASCENDING_THRESHOLD:
                state = "ascending"
                candidate_count = 0

        # ASCENDING: confirm return to top
        elif state == "ascending":

            if angle < bottom_angle:
                bottom_angle = angle
                bottom_time = timestamp

            if angle < ASCENDING_THRESHOLD:
                state = "bottom"
                candidate_count = 0

            elif angle >= TOP_THRESHOLD:

                candidate_count += 1

                if candidate_count >= CONFIRMATION_FRAMES:

                    end_time = timestamp
                    rep_number += 1

                    reps.append({
                        "rep_number": rep_number,
                        "start_time": safe_round(start_time),
                        "bottom_time": safe_round(bottom_time),
                        "end_time": safe_round(end_time),
                        "bottom_elbow_angle": safe_round(bottom_angle),
                        "rep_time": safe_round(
                            end_time - start_time
                        ),
                    })

                    state = "top"
                    candidate_count = 0

                    start_time = None
                    candidate_start_time = None
                    bottom_angle = None
                    bottom_time = None

            else:
                candidate_count = 0

    return reps


def analyze_pushup(input_csv, output_json):

    input_csv = Path(input_csv)
    output_json = Path(output_json)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required = [
        "time_seconds",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_body_alignment",
        "right_body_alignment",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    for column in required:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = (
        df.sort_values("time_seconds")
        .drop_duplicates("time_seconds")
        .reset_index(drop=True)
    )

    if len(df) < MIN_VALID_FRAMES:
        raise ValueError("Not enough frames for push-up analysis.")

    total_frames = len(df)

    # ---------------------------------------------------------
    # MEASUREMENT COVERAGE
    # ---------------------------------------------------------

    left_elbow_frames = int(df["left_elbow_angle"].notna().sum())
    right_elbow_frames = int(df["right_elbow_angle"].notna().sum())

    left_body_frames = int(df["left_body_alignment"].notna().sum())
    right_body_frames = int(df["right_body_alignment"].notna().sum())

    paired_elbows = df[
        ["left_elbow_angle", "right_elbow_angle"]
    ].dropna()

    paired_body = df[
        ["left_body_alignment", "right_body_alignment"]
    ].dropna()

    elbow_coverage = max(
        left_elbow_frames, right_elbow_frames
    ) / total_frames

    alignment_coverage = max(
        left_body_frames, right_body_frames
    ) / total_frames

    symmetry_coverage = len(paired_elbows) / total_frames

    alignment_symmetry_coverage = len(paired_body) / total_frames

    elbow_available = (
        max(left_elbow_frames, right_elbow_frames)
        >= MIN_VALID_FRAMES
        and elbow_coverage >= MIN_MEASUREMENT_COVERAGE
    )

    alignment_available = (
        max(left_body_frames, right_body_frames)
        >= MIN_VALID_FRAMES
        and alignment_coverage >= MIN_MEASUREMENT_COVERAGE
    )

    symmetry_available = (
        len(paired_elbows) >= MIN_VALID_FRAMES
        and symmetry_coverage >= MIN_MEASUREMENT_COVERAGE
    )

    # ---------------------------------------------------------
    # CHOOSE THE MOST RELIABLE ELBOW FOR REP DETECTION
    # ---------------------------------------------------------

    primary_elbow = (
        "left_elbow_angle"
        if left_elbow_frames >= right_elbow_frames
        else "right_elbow_angle"
    )

    df["primary_elbow_angle"] = df[primary_elbow]

    df["primary_elbow_angle"] = (
        df["primary_elbow_angle"]
        .rolling(
            SMOOTHING_WINDOW,
            min_periods=1,
            center=True,
        )
        .median()
    )

    # Average both elbows only when both are present.
    # Otherwise use the primary elbow for graph continuity.
    df["average_elbow_angle"] = df[
        ["left_elbow_angle", "right_elbow_angle"]
    ].mean(axis=1)

    df["average_elbow_angle"] = df[
        "average_elbow_angle"
    ].fillna(df["primary_elbow_angle"])

    df["average_elbow_angle"] = (
        df["average_elbow_angle"]
        .rolling(
            SMOOTHING_WINDOW,
            min_periods=1,
            center=True,
        )
        .median()
    )

    # ---------------------------------------------------------
    # BODY ALIGNMENT
    # ---------------------------------------------------------

    df["average_body_alignment"] = df[
        ["left_body_alignment", "right_body_alignment"]
    ].mean(axis=1)

    df["average_body_alignment"] = (
        df["average_body_alignment"]
        .rolling(
            SMOOTHING_WINDOW,
            min_periods=1,
            center=True,
        )
        .median()
    )

    # ---------------------------------------------------------
    # REPETITION DETECTION
    # ---------------------------------------------------------

    reps = calculate_repetitions(
        df,
        "primary_elbow_angle",
    )

    repetitions = len(reps)

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    elbow_data = df["average_elbow_angle"].dropna()

    alignment_data = df[
        "average_body_alignment"
    ].dropna()

    elbow_rom = (
        float(elbow_data.max() - elbow_data.min())
        if elbow_available and len(elbow_data)
        else None
    )

    symmetry_difference = (
        float(
            (
                paired_elbows["left_elbow_angle"]
                - paired_elbows["right_elbow_angle"]
            ).abs().mean()
        )
        if symmetry_available
        else None
    )

    alignment_deviation = (
        float((180 - alignment_data).abs().mean())
        if alignment_available and len(alignment_data)
        else None
    )

    rep_times = [
        rep["rep_time"]
        for rep in reps
        if rep["rep_time"] is not None
    ]

    average_rep_time = (
        sum(rep_times) / len(rep_times)
        if rep_times
        else None
    )

    rep_time_difference = (
        max(rep_times) - min(rep_times)
        if len(rep_times) >= 2
        else None
    )

    bottom_angles = [
        rep["bottom_elbow_angle"]
        for rep in reps
        if rep["bottom_elbow_angle"] is not None
    ]

    average_bottom_angle = (
        sum(bottom_angles) / len(bottom_angles)
        if bottom_angles
        else None
    )

    # ---------------------------------------------------------
    # COMPONENT SCORES
    # ---------------------------------------------------------

    depth_score = None
    alignment_score = None
    symmetry_score = None
    tempo_score = None
    consistency_score = None

    if repetitions > 0:

        if average_bottom_angle is not None:

            if average_bottom_angle <= 75:
                depth_score = 100
            elif average_bottom_angle <= 90:
                depth_score = 85
            elif average_bottom_angle <= 105:
                depth_score = 70
            else:
                depth_score = 50

        if alignment_deviation is not None:

            if alignment_deviation <= 5:
                alignment_score = 100
            elif alignment_deviation <= 10:
                alignment_score = 85
            elif alignment_deviation <= 20:
                alignment_score = 70
            else:
                alignment_score = 50

        if symmetry_difference is not None:

            if symmetry_difference <= 5:
                symmetry_score = 100
            elif symmetry_difference <= 10:
                symmetry_score = 85
            elif symmetry_difference <= 20:
                symmetry_score = 70
            else:
                symmetry_score = 50

        if average_rep_time is not None:

            if 1.5 <= average_rep_time <= 4:
                tempo_score = 100
            elif 1 <= average_rep_time <= 5:
                tempo_score = 85
            else:
                tempo_score = 70

        if rep_time_difference is not None:

            if rep_time_difference <= 0.20:
                consistency_score = 100
            elif rep_time_difference <= 0.40:
                consistency_score = 85
            elif rep_time_difference <= 0.70:
                consistency_score = 70
            else:
                consistency_score = 50

    component_scores = {
        "depth_score": depth_score,
        "alignment_score": alignment_score,
        "symmetry_score": symmetry_score,
        "tempo_score": tempo_score,
        "consistency_score": consistency_score,
    }

    available = [
        name for name, score in component_scores.items()
        if score is not None
    ]

    unavailable = [
        name for name, score in component_scores.items()
        if score is None
    ]

    available_weight = sum(
        COMPONENT_WEIGHTS[name]
        for name in available
    )

    coverage = round(available_weight * 100, 2)

    performance_score = (
        round(
            sum(
                component_scores[name]
                * COMPONENT_WEIGHTS[name]
                for name in available
            ) / available_weight,
            2,
        )
        if repetitions > 0 and available_weight > 0
        else None
    )

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    if repetitions == 0 or performance_score is None:
        status = "Insufficient Data"
    elif coverage < 100:
        status = "Provisional"
    else:
        status = "Complete"

    if performance_score is None:
        band = "Insufficient Data"
    elif performance_score >= 85:
        band = "Strong"
    elif performance_score >= 70:
        band = "Good"
    elif performance_score >= 55:
        band = "Developing"
    else:
        band = "Needs Improvement"

    movement_level = (
        "Insufficient Data"
        if status == "Insufficient Data"
        else "Provisional"
        if status == "Provisional"
        else band
    )

    # ---------------------------------------------------------
    # RECOMMENDATIONS
    # ---------------------------------------------------------

    recommendations = []

    if repetitions == 0:

        recommendations.append(
            "No complete push-up repetitions were detected. "
            "Ensure the movement and your upper body are visible."
        )

    else:

        if depth_score is None:
            recommendations.append(
                "Push-up depth could not be assessed reliably."
            )
        elif depth_score < 85:
            recommendations.append(
                "Work on maintaining a consistent elbow movement range."
            )
        else:
            recommendations.append(
                "Good elbow depth was observed in the detected repetitions."
            )

        if alignment_score is None:
            recommendations.append(
                "Body alignment could not be assessed reliably."
            )
        elif alignment_score < 85:
            recommendations.append(
                "Focus on maintaining a straighter body line."
            )
        else:
            recommendations.append(
                "Body alignment remained relatively consistent."
            )

        if symmetry_score is None:
            recommendations.append(
                "Elbow symmetry could not be assessed reliably "
                "because both elbows were not sufficiently visible."
            )
        elif symmetry_score < 85:
            recommendations.append(
                "Focus on keeping left-right elbow movement consistent."
            )
        else:
            recommendations.append(
                "Left-right elbow movement was reasonably consistent."
            )

        if tempo_score is None:
            recommendations.append(
                "Movement tempo could not be assessed reliably."
            )
        else:
            recommendations.append(
                "Repetition duration was used to assess movement tempo."
            )

        if consistency_score is None:
            recommendations.append(
                "More complete repetitions are needed to assess "
                "repetition timing consistency."
            )
        elif consistency_score < 85:
            recommendations.append(
                "Try to maintain more consistent duration across repetitions."
            )
        else:
            recommendations.append(
                "Repetition timing was reasonably consistent."
            )

    if status == "Provisional":
        recommendations.append(
            "This assessment is provisional because some "
            "measurement components were unavailable."
        )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    assessment = {
        "activity": "pushup",
        "performance_score": performance_score,
        "movement_level": movement_level,
        "performance_band": band,
        "assessment_status": status,

        "score_coverage_percent": coverage,
        "available_component_count": len(available),
        "total_component_count": len(COMPONENT_WEIGHTS),

        "available_components": available,
        "unavailable_components": unavailable,

        "repetitions": repetitions,

        "metrics": {
            "elbow_range_of_motion": safe_round(elbow_rom),
            "average_bottom_elbow_angle": safe_round(
                average_bottom_angle
            ),
            "body_alignment_deviation": safe_round(
                alignment_deviation
            ),
            "elbow_symmetry_difference": safe_round(
                symmetry_difference
            ),
            "elbow_symmetry_available": symmetry_available,
            "average_rep_time": safe_round(average_rep_time),
            "rep_time_difference": safe_round(
                rep_time_difference
            ),
        },

        "measurement_coverage": {
            "elbow_coverage_percent": round(
                elbow_coverage * 100, 2
            ),
            "alignment_coverage_percent": round(
                alignment_coverage * 100, 2
            ),
            "symmetry_coverage_percent": round(
                symmetry_coverage * 100, 2
            ),
            "alignment_symmetry_coverage_percent": round(
                alignment_symmetry_coverage * 100, 2
            ),
            "primary_elbow": (
                "left" if primary_elbow == "left_elbow_angle"
                else "right"
            ),
        },

        "component_scores": component_scores,
        "rep_details": reps,
        "recommendations": recommendations,
    }

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(assessment, file, indent=4)

    print("\nPUSH-UP ANALYSIS COMPLETED")
    print("Repetitions:", repetitions)
    print("Performance score:", performance_score)
    print("Performance band:", band)
    print("Assessment status:", status)
    print("Score coverage:", coverage, "%")
    print("Available components:", len(available))
    print("Unavailable components:", len(unavailable))
    print("Output JSON:", output_json)

    return assessment


if __name__ == "__main__":

    analyze_pushup(
        "data/processed/pushup_features.csv",
        "data/processed/pushup_features.json",
    )
