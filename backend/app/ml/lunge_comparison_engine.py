
import numpy as np
import pandas as pd


def _number(value):
    """Convert a value to a finite float, or return None."""
    try:
        if value is None:
            return None

        number = float(value)

        if not np.isfinite(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


def _metric_comparison(
    label,
    value_a,
    value_b,
    unit="",
    direction="context"
):
    """
    Compare one metric between Video A and Video B.

    direction:
        higher  -> higher value is better
        lower   -> lower value is better
        context -> change depends on exercise context
    """

    value_a = _number(value_a)
    value_b = _number(value_b)

    if value_a is None or value_b is None:
        return {
            "label": label,
            "video_a": value_a,
            "video_b": value_b,
            "difference": None,
            "unit": unit,
            "direction": direction,
            "interpretation": "Unavailable"
        }

    difference = value_b - value_a

    if abs(difference) < 0.001:
        interpretation = "No Significant Change"

    elif direction == "higher":
        interpretation = (
            "Improved" if difference > 0 else "Declined"
        )

    elif direction == "lower":
        interpretation = (
            "Improved" if difference < 0 else "Declined"
        )

    else:
        interpretation = "Changed"

    return {
        "label": label,
        "video_a": round(value_a, 2),
        "video_b": round(value_b, 2),
        "difference": round(difference, 2),
        "unit": unit,
        "direction": direction,
        "interpretation": interpretation
    }


def _get_selected_knee_column(assessment):
    """Choose the knee-angle column based on the selected side."""

    selected_side = str(
        assessment.get("selected_side", "")
    ).strip().lower()

    if selected_side == "left":
        return "left_knee_angle"

    if selected_side == "right":
        return "right_knee_angle"

    return None


def _build_movement_curve(
    csv_path,
    assessment,
    points=101
):
    """
    Build an average knee-angle movement curve.

    Each repetition is normalized independently to 0–100%
    before calculating the average curve.
    """

    selected_side = assessment.get("selected_side")
    knee_column = _get_selected_knee_column(assessment)

    unavailable = {
        "available": False,
        "selected_side": selected_side,
        "progress": [],
        "values": [],
        "repetitions_used": 0,
        "description": "Movement curve could not be generated."
    }

    if not knee_column:
        return unavailable

    try:
        df = pd.read_csv(csv_path)

        required_columns = [
            "time_seconds",
            knee_column
        ]

        if not all(
            column in df.columns
            for column in required_columns
        ):
            return unavailable

        df = df[required_columns].copy()

        df["time_seconds"] = pd.to_numeric(
            df["time_seconds"],
            errors="coerce"
        )

        df[knee_column] = pd.to_numeric(
            df[knee_column],
            errors="coerce"
        )

        df = df.dropna()

        df = df[
            np.isfinite(df["time_seconds"])
            & np.isfinite(df[knee_column])
        ]

        df = df.sort_values("time_seconds")

        if len(df) < 2:
            return unavailable

        repetitions = assessment.get("repetitions", [])

        if not isinstance(repetitions, list) or not repetitions:
            return unavailable

        normalized_repetitions = []

        for repetition in repetitions:

            start_time = _number(
                repetition.get("start_time")
            )

            end_time = _number(
                repetition.get("end_time")
            )

            if (
                start_time is None
                or end_time is None
                or end_time <= start_time
            ):
                continue

            rep_df = df[
                (df["time_seconds"] >= start_time)
                & (df["time_seconds"] <= end_time)
            ]

            if len(rep_df) < 2:
                continue

            times = rep_df["time_seconds"].to_numpy(
                dtype=float
            )

            angles = rep_df[knee_column].to_numpy(
                dtype=float
            )

            # Remove duplicate timestamps.
            unique_times, unique_indices = np.unique(
                times,
                return_index=True
            )

            angles = angles[unique_indices]
            times = unique_times

            if len(times) < 2:
                continue

            duration = times[-1] - times[0]

            if duration <= 0:
                continue

            normalized_time = (
                (times - times[0]) / duration
            )

            target_progress = np.linspace(
                0,
                1,
                points
            )

            normalized_angles = np.interp(
                target_progress,
                normalized_time,
                angles
            )

            normalized_repetitions.append(
                normalized_angles
            )

        if not normalized_repetitions:
            return unavailable

        average_curve = np.mean(
            normalized_repetitions,
            axis=0
        )

        progress = np.linspace(
            0,
            100,
            points
        )

        return {
            "available": True,
            "selected_side": selected_side,
            "progress": [
                round(float(value), 2)
                for value in progress
            ],
            "values": [
                round(float(value), 2)
                for value in average_curve
            ],
            "repetitions_used": len(
                normalized_repetitions
            ),
            "description": (
                "Average knee-angle curve calculated "
                "from normalized repetitions."
            )
        }

    except Exception:
        return unavailable


def _compare_curves(curve_a, curve_b):
    """Compare two normalized movement curves."""

    if (
        not curve_a.get("available")
        or not curve_b.get("available")
    ):
        return {
            "available": False,
            "progress": [],
            "video_a": [],
            "video_b": [],
            "mean_absolute_difference": None,
            "unit": "degrees",
            "repetitions_used": {
                "video_a": curve_a.get(
                    "repetitions_used", 0
                ),
                "video_b": curve_b.get(
                    "repetitions_used", 0
                )
            },
            "selected_sides": {
                "video_a": curve_a.get("selected_side"),
                "video_b": curve_b.get("selected_side")
            },
            "description": (
                "Movement curve comparison is unavailable "
                "because sufficient repetition data was "
                "not available for both videos."
            )
        }

    values_a = np.array(
        curve_a["values"],
        dtype=float
    )

    values_b = np.array(
        curve_b["values"],
        dtype=float
    )

    if (
        len(values_a) == 0
        or len(values_a) != len(values_b)
    ):
        return {
            "available": False,
            "progress": [],
            "video_a": [],
            "video_b": [],
            "mean_absolute_difference": None,
            "unit": "degrees",
            "repetitions_used": {
                "video_a": curve_a.get(
                    "repetitions_used", 0
                ),
                "video_b": curve_b.get(
                    "repetitions_used", 0
                )
            },
            "selected_sides": {
                "video_a": curve_a.get("selected_side"),
                "video_b": curve_b.get("selected_side")
            },
            "description": (
                "Movement curves could not be aligned."
            )
        }

    mean_absolute_difference = float(
        np.mean(np.abs(values_b - values_a))
    )

    return {
        "available": True,
        "progress": curve_a["progress"],
        "video_a": curve_a["values"],
        "video_b": curve_b["values"],
        "mean_absolute_difference": round(
            mean_absolute_difference,
            2
        ),
        "unit": "degrees",
        "repetitions_used": {
            "video_a": curve_a["repetitions_used"],
            "video_b": curve_b["repetitions_used"]
        },
        "selected_sides": {
            "video_a": curve_a.get("selected_side"),
            "video_b": curve_b.get("selected_side")
        },
        "description": (
            "Both movement curves are normalized to "
            "0–100% movement progress. The difference "
            "represents the mean absolute knee-angle "
            "difference between the two average curves."
        )
    }


def compare_lunge_results(
    assessment_a,
    assessment_b,
    csv_path_a,
    csv_path_b,
    comparison_mode="progress"
):
    """
    Compare two Lunge assessments.

    comparison_mode:
        progress  -> compare the athlete's two performances
        reference -> compare the athlete against a reference video
    """

    if comparison_mode not in ["progress", "reference"]:
        raise ValueError(
            "comparison_mode must be 'progress' or 'reference'."
        )

    # --------------------------------------------------
    # Metric comparisons
    # --------------------------------------------------

    metrics = {
        "average_knee_rom": _metric_comparison(
            "Average Knee Range of Motion",
            assessment_a.get("average_knee_rom"),
            assessment_b.get("average_knee_rom"),
            "degrees",
            "context"
        ),

        "minimum_knee_angle": _metric_comparison(
            "Minimum Knee Angle",
            assessment_a.get("minimum_knee_angle"),
            assessment_b.get("minimum_knee_angle"),
            "degrees",
            "context"
        ),

        "average_knee_angle": _metric_comparison(
            "Average Knee Angle",
            assessment_a.get("average_knee_angle"),
            assessment_b.get("average_knee_angle"),
            "degrees",
            "context"
        ),

        "average_rep_duration": _metric_comparison(
            "Average Repetition Duration",
            assessment_a.get("average_rep_duration"),
            assessment_b.get("average_rep_duration"),
            "seconds",
            "context"
        ),

        "rep_consistency": _metric_comparison(
            "Repetition Consistency",
            assessment_a.get("rep_consistency"),
            assessment_b.get("rep_consistency"),
            "%",
            "higher"
        ),

        "average_torso_angle": _metric_comparison(
            "Average Torso Angle",
            assessment_a.get("average_torso_angle"),
            assessment_b.get("average_torso_angle"),
            "degrees",
            "context"
        ),

        "coverage": _metric_comparison(
            "Pose Visibility Coverage",
            assessment_a.get("coverage"),
            assessment_b.get("coverage"),
            "%",
            "higher"
        )
    }

    # --------------------------------------------------
    # Performance score comparison
    # --------------------------------------------------

    score_a = _number(
        assessment_a.get("performance_score")
    )

    score_b = _number(
        assessment_b.get("performance_score")
    )

    score_difference = None

    if score_a is not None and score_b is not None:

        score_difference = round(
            score_b - score_a,
            2
        )

        if abs(score_difference) < 0.001:
            score_interpretation = "No Significant Change"

        elif score_difference > 0:
            score_interpretation = "Improved"

        else:
            score_interpretation = "Declined"

    else:
        score_interpretation = "Unavailable"

    score_comparison = {
        "video_a": score_a,
        "video_b": score_b,
        "difference": score_difference,
        "interpretation": score_interpretation
    }

    # --------------------------------------------------
    # Repetition comparison
    # --------------------------------------------------

    reps_a = assessment_a.get(
        "detected_repetitions"
    )

    reps_b = assessment_b.get(
        "detected_repetitions"
    )

    reps_a = (
        int(reps_a)
        if reps_a is not None
        else None
    )

    reps_b = (
        int(reps_b)
        if reps_b is not None
        else None
    )

    repetition_difference = None

    if reps_a is not None and reps_b is not None:
        repetition_difference = reps_b - reps_a

    repetitions_comparison = {
        "video_a": reps_a,
        "video_b": reps_b,
        "difference": repetition_difference,
        "note": (
            "A higher repetition count does not necessarily "
            "indicate better performance. Repetition quality, "
            "depth, control and consistency should also "
            "be considered."
        )
    }

    # --------------------------------------------------
    # Movement curve comparison
    # --------------------------------------------------

    curve_a = _build_movement_curve(
        csv_path_a,
        assessment_a
    )

    curve_b = _build_movement_curve(
        csv_path_b,
        assessment_b
    )

    movement_curve_comparison = _compare_curves(
        curve_a,
        curve_b
    )

    # --------------------------------------------------
    # Observations
    # --------------------------------------------------

    observations = []

    if score_interpretation == "Improved":
        observations.append(
            "The overall performance score is higher in Video B."
        )

    elif score_interpretation == "Declined":
        observations.append(
            "The overall performance score is lower in Video B."
        )

    elif score_interpretation == "No Significant Change":
        observations.append(
            "The overall performance scores are nearly the same."
        )

    if metrics["rep_consistency"]["interpretation"] == "Improved":
        observations.append(
            "Repetition consistency improved in Video B."
        )

    elif metrics["rep_consistency"]["interpretation"] == "Declined":
        observations.append(
            "Repetition consistency declined in Video B."
        )

    # Corrected coverage observation.
    if metrics["coverage"]["interpretation"] == "Improved":
        observations.append(
            "Pose visibility coverage is higher in Video B."
        )

    elif metrics["coverage"]["interpretation"] == "Declined":
        observations.append(
            "Pose visibility coverage is lower in Video B."
        )

    if (
        reps_a is not None
        and reps_b is not None
        and reps_a != reps_b
    ):
        observations.append(
            f"Video B contains {reps_b} detected repetitions "
            f"compared with {reps_a} in Video A. "
            "Repetition count alone does not determine quality."
        )

    if (
        curve_a.get("available")
        and curve_b.get("available")
    ):
        observations.append(
            "The normalized knee-angle movement patterns "
            "are available for visual comparison."
        )

    if not observations:
        observations.append(
            "No major differences were identified from "
            "the available comparison metrics."
        )

    # --------------------------------------------------
    # Limitations
    # --------------------------------------------------

    limitations = [
        (
            "Lunge analysis is based on 2D pose estimation. "
            "Camera angle, body visibility and lighting can "
            "affect landmark accuracy."
        ),
        (
            "Average movement curves can hide differences "
            "between individual repetitions."
        ),
        (
            "Knee range of motion and repetition duration "
            "depend on exercise technique and context. "
            "Higher or lower values are not automatically better."
        ),
        (
            "The performance score is a prototype assessment "
            "and has not been scientifically validated."
        )
    ]

    selected_side_a = assessment_a.get("selected_side")
    selected_side_b = assessment_b.get("selected_side")

    if (
        selected_side_a
        and selected_side_b
        and str(selected_side_a).lower()
        != str(selected_side_b).lower()
    ):
        limitations.append(
            "Different legs were selected for the two videos "
            f"({selected_side_a} in Video A and "
            f"{selected_side_b} in Video B). This may affect "
            "direct movement comparisons."
        )

    coverage_a = _number(
        assessment_a.get("coverage")
    )

    coverage_b = _number(
        assessment_b.get("coverage")
    )

    if (
        coverage_a is not None
        and coverage_b is not None
        and abs(coverage_b - coverage_a) >= 10
    ):
        limitations.append(
            "Pose visibility coverage differs by at least "
            "10 percentage points between the videos. "
            "This may affect comparison reliability."
        )

    status_a = assessment_a.get("assessment_status")
    status_b = assessment_b.get("assessment_status")

    if status_a and status_b and status_a != status_b:
        limitations.append(
            "The two videos have different assessment "
            "statuses, so their scores should be interpreted "
            "with caution."
        )

    if status_a == "Limited" or status_b == "Limited":
        limitations.append(
            "At least one video has a limited assessment. "
            "Its results may not contain enough reliable "
            "information for a complete comparison."
        )

    if not movement_curve_comparison["available"]:
        limitations.append(
            "Movement curve comparison is unavailable because "
            "sufficient valid repetition data was not found "
            "for both videos."
        )

    # --------------------------------------------------
    # Final response
    # --------------------------------------------------

    return {
        "success": True,
        "exercise": "Lunge",
        "comparison_mode": comparison_mode,

        "video_a": {
            "performance_score": score_a,
            "performance_status": assessment_a.get(
                "performance_status"
            ),
            "assessment_status": status_a,
            "detected_repetitions": reps_a,
            "selected_side": selected_side_a
        },

        "video_b": {
            "performance_score": score_b,
            "performance_status": assessment_b.get(
                "performance_status"
            ),
            "assessment_status": status_b,
            "detected_repetitions": reps_b,
            "selected_side": selected_side_b
        },

        "score_comparison": score_comparison,

        "repetitions_comparison": repetitions_comparison,

        "metric_comparisons": metrics,

        "movement_curve_comparison": (
            movement_curve_comparison
        ),

        "observations": observations,

        "limitations": limitations
    }


# Backward-compatible alias.
compare_analyses = compare_lunge_results
