
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from backend.app.ml.comparison_phase_alignment import (
        align_squat_curves
    )
except ImportError:
    align_squat_curves = None


SMOOTHING_WINDOW = 5
DEFAULT_CURVE_POINTS = 101


def extract_normalized_knee_curve(
    csv_path,
    points=DEFAULT_CURVE_POINTS
):
    """
    Extract average knee angle and normalize the video timeline
    to 0-100 percent.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Angle CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if "time_seconds" not in df.columns:
        raise ValueError(
            "CSV must contain time_seconds."
        )

    knee_columns = [
        column
        for column in [
            "left_knee_angle",
            "right_knee_angle"
        ]
        if column in df.columns
    ]

    if not knee_columns:
        raise ValueError(
            "CSV contains no knee-angle columns."
        )

    df["time_seconds"] = pd.to_numeric(
        df["time_seconds"],
        errors="coerce"
    )

    for column in knee_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["average_knee_angle"] = df[
        knee_columns
    ].mean(axis=1)

    df = df[
        ["time_seconds", "average_knee_angle"]
    ].dropna()

    df = df.sort_values("time_seconds")

    df = df.drop_duplicates(
        subset=["time_seconds"],
        keep="first"
    )

    if len(df) < 2:
        raise ValueError(
            "Not enough valid knee-angle data."
        )

    df["average_knee_angle"] = (
        df["average_knee_angle"]
        .rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1,
            center=True
        )
        .mean()
    )

    start_time = df["time_seconds"].min()
    end_time = df["time_seconds"].max()

    if end_time <= start_time:
        raise ValueError(
            "Invalid video timeline."
        )

    original_percent = (
        (df["time_seconds"] - start_time)
        / (end_time - start_time)
    ) * 100

    target_percent = np.linspace(
        0,
        100,
        points
    )

    normalized_angles = np.interp(
        target_percent,
        original_percent.to_numpy(),
        df["average_knee_angle"].to_numpy()
    )

    return [
        {
            "time_percent": round(
                float(time), 2
            ),
            "knee_angle": round(
                float(angle), 2
            )
        }
        for time, angle in zip(
            target_percent,
            normalized_angles
        )
    ]


def _flatten_numeric_metrics(data, prefix=""):
    """
    Extract comparable numeric assessment values.
    """

    result = {}

    if not isinstance(data, dict):
        return result

    ignored_keys = {
        "rep_details",
        "repetitions_details",
        "raw_data",
        "metadata",
        "coverage",
        "visibility",
    }

    for key, value in data.items():

        key = str(key)

        if key.lower() in ignored_keys:
            continue

        full_key = (
            f"{prefix}.{key}"
            if prefix
            else key
        )

        if isinstance(value, dict):
            result.update(
                _flatten_numeric_metrics(
                    value,
                    full_key
                )
            )

        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
        ):
            result[full_key] = float(value)

    return result


def _metric_interpretation(metric_name, difference):
    """
    Do not assume that every increase means improvement.
    """

    name = metric_name.lower()

    higher_is_better = [
        "performance_score",
        "overall_score",
        "form_score",
        "technique_score",
        "symmetry_score",
        "depth_score",
    ]

    if any(
        name.endswith(metric)
        for metric in higher_is_better
    ):
        if difference > 0:
            return "Higher in video B"

        if difference < 0:
            return "Higher in video A"

        return "No difference"

    return "Difference detected; interpretation depends on context"


def _normalize_curve(curve):
    """
    Accept normalized curves and standardize their values.
    """

    if not isinstance(curve, list):
        return []

    normalized = []

    for item in curve:
        if not isinstance(item, dict):
            continue

        time = item.get("time_percent")
        angle = item.get("knee_angle")

        if angle is None:
            angle = item.get("average_knee_angle")

        try:
            time = float(time)
            angle = float(angle)
        except (TypeError, ValueError):
            continue

        if not (
            np.isfinite(time)
            and np.isfinite(angle)
        ):
            continue

        normalized.append({
            "time_percent": time,
            "knee_angle": angle
        })

    return sorted(
        normalized,
        key=lambda item: item["time_percent"]
    )


def _compare_curves(curve_a, curve_b):
    """
    Compare two curves using normalized full-video timelines.

    This is descriptive timeline comparison, not movement-phase alignment.
    """

    curve_a = _normalize_curve(curve_a)
    curve_b = _normalize_curve(curve_b)

    if len(curve_a) < 2 or len(curve_b) < 2:
        return []

    x_a = np.array([
        item["time_percent"]
        for item in curve_a
    ])

    y_a = np.array([
        item["knee_angle"]
        for item in curve_a
    ])

    x_b = np.array([
        item["time_percent"]
        for item in curve_b
    ])

    y_b = np.array([
        item["knee_angle"]
        for item in curve_b
    ])

    timeline = np.linspace(0, 100, 101)

    values_a = np.interp(
        timeline,
        x_a,
        y_a
    )

    values_b = np.interp(
        timeline,
        x_b,
        y_b
    )

    result = []

    for time, angle_a, angle_b in zip(
        timeline,
        values_a,
        values_b
    ):
        result.append({
            "time_percent": round(
                float(time), 2
            ),
            "knee_angle_a": round(
                float(angle_a), 2
            ),
            "knee_angle_b": round(
                float(angle_b), 2
            ),
            "difference": round(
                float(angle_b - angle_a), 2
            )
        })

    return result


def _build_phase_alignment(curve_a, curve_b):
    """
    Build separate movement-phase alignment data.

    Does not replace the existing full-video curve comparison.
    """

    if align_squat_curves is None:
        return {
            "available": False,
            "message": (
                "Phase alignment module is unavailable. "
                "The full-video curve comparison is still available."
            ),
            "video_a": None,
            "video_b": None,
            "aligned_curve": []
        }

    try:
        return align_squat_curves(
            curve_a,
            curve_b
        )

    except Exception as error:
        return {
            "available": False,
            "message": (
                "Phase alignment could not be completed: "
                f"{str(error)}"
            ),
            "video_a": None,
            "video_b": None,
            "aligned_curve": []
        }


def build_squat_comparison(
    assessment_a,
    assessment_b,
    curve_a,
    curve_b,
    comparison_mode="progress"
):
    """
    Build a comparison result without modifying
    either video's original assessment.
    """

    if comparison_mode not in [
        "progress",
        "reference"
    ]:
        raise ValueError(
            "comparison_mode must be progress or reference."
        )

    metrics_a = _flatten_numeric_metrics(
        assessment_a
    )

    metrics_b = _flatten_numeric_metrics(
        assessment_b
    )

    metrics = []

    common_keys = sorted(
        set(metrics_a) & set(metrics_b)
    )

    for key in common_keys:

        value_a = metrics_a[key]
        value_b = metrics_b[key]

        difference = value_b - value_a

        metrics.append({
            "metric": key,
            "value_a": round(value_a, 2),
            "value_b": round(value_b, 2),
            "difference": round(
                difference, 2
            ),
            "interpretation": (
                _metric_interpretation(
                    key,
                    difference
                )
            )
        })

    # Existing full-video timeline comparison.
    curve_comparison = _compare_curves(
        curve_a,
        curve_b
    )

    # New movement-phase comparison.
    phase_alignment = _build_phase_alignment(
        curve_a,
        curve_b
    )

    observations = []

    if not metrics:
        observations.append(
            "No common numeric assessment metrics were available."
        )

    if not curve_comparison:
        observations.append(
            "Insufficient knee-angle data for full-video curve comparison."
        )

    if metrics:
        observations.append(
            "Numeric assessment values were compared between the two videos."
        )

    if curve_comparison:
        observations.append(
            "Knee-angle curves were compared on a normalized full-video timeline. "
            "This does not guarantee matching movement phases."
        )

    if phase_alignment.get("available"):
        observations.append(
            "Detectable squat movement attempts were separately aligned "
            "across descent, bottom, and ascent phases."
        )
    else:
        observations.append(
            "Movement-phase alignment was unavailable. "
            "The full-video curve remains descriptive only."
        )

    limitations = [
        "Video comparison depends on pose-detection quality.",
        "Camera angle, visibility, and body positioning can affect measured angles.",
        "A numeric difference does not automatically indicate better technique.",
        "Only metrics available in both assessments are compared.",
        "Full-video timeline alignment does not guarantee corresponding movement phases.",
        "Phase alignment identifies movement attempts and does not replace the original repetition counter.",
        "Incomplete movement attempts must not be treated as completed repetitions.",
        "The comparison does not replace professional coaching or medical assessment."
    ]

    return {
        "activity": "Squat Comparison",
        "comparison_mode": comparison_mode,
        "title": (
            "Squat Progress Comparison"
            if comparison_mode == "progress"
            else "Squat Reference Comparison"
        ),
        "metrics": metrics,
        "curve_comparison": curve_comparison,
        "phase_alignment": phase_alignment,
        "observations": observations,
        "limitations": limitations
    }
