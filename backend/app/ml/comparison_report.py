
# backend/app/ml/comparison_report.py

from typing import Any, Dict, List, Optional


# Only athlete-relevant metrics are included in the comparison report.
METRIC_CONFIG = {
    "knee_range_of_motion": {
        "label": "Knee Range of Motion",
        "unit": "°",
        "meaning": "Describes the measured knee movement range during the video.",
        "interpretation": (
            "A larger range indicates more measured knee movement, "
            "but does not automatically mean better squat performance."
        ),
        "direction": "context",
    },
    "hip_range_of_motion": {
        "label": "Hip Range of Motion",
        "unit": "°",
        "meaning": "Describes the measured hip movement range during the video.",
        "interpretation": (
            "A larger range indicates more measured hip movement, "
            "but does not automatically mean better squat performance."
        ),
        "direction": "context",
    },
    "knee_symmetry_difference": {
        "label": "Knee Symmetry Difference",
        "unit": "°",
        "meaning": "Measures the difference between left and right knee angles.",
        "interpretation": (
            "A smaller difference indicates closer left-right similarity."
        ),
        "direction": "lower",
    },
    "hip_symmetry_difference": {
        "label": "Hip Symmetry Difference",
        "unit": "°",
        "meaning": "Measures the difference between left and right hip angles.",
        "interpretation": (
            "A smaller difference indicates closer left-right similarity."
        ),
        "direction": "lower",
    },
    "trunk_angle_change": {
        "label": "Trunk Angle Change",
        "unit": "°",
        "meaning": "Describes the measured change in trunk angle.",
        "interpretation": (
            "Its significance depends on movement quality and intended technique."
        ),
        "direction": "context",
    },
}


def _number(value: Any) -> Optional[float]:
    """Convert a value to a finite float, otherwise return None."""
    try:
        if value is None:
            return None

        result = float(value)

        if result != result or result in (float("inf"), float("-inf")):
            return None

        return round(result, 2)

    except (TypeError, ValueError):
        return None


def _get_metrics(assessment: Dict[str, Any]) -> Dict[str, Any]:
    return assessment.get("metrics") or {}


def _compare_metric(
    key: str,
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    config = METRIC_CONFIG[key]

    value_a = _number(metrics_a.get(key))
    value_b = _number(metrics_b.get(key))

    if value_a is None or value_b is None:
        return None

    difference = round(value_b - value_a, 2)
    direction = config["direction"]

    if difference == 0:
        interpretation = "No measurable difference was detected."

    elif direction == "lower":
        if difference < 0:
            interpretation = (
                f"{config['label']} is smaller in Video B. "
                "This indicates closer left-right similarity for this metric."
            )
        else:
            interpretation = (
                f"{config['label']} is larger in Video B. "
                "This indicates less left-right similarity for this metric."
            )

    elif direction == "context":
        interpretation = config["interpretation"]

    return {
        "key": key,
        "label": config["label"],
        "video_a": value_a,
        "video_b": value_b,
        "difference": difference,
        "unit": config["unit"],
        "interpretation": interpretation,
        "meaning": config["meaning"],
    }


def _movement_diagnostics(
    assessment: Dict[str, Any],
    curve: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:

    repetitions = assessment.get("repetitions", 0)
    repetitions = repetitions if isinstance(repetitions, int) else 0

    angles = []

    for point in curve or []:
        angle = _number(point.get("knee_angle"))
        if angle is not None:
            angles.append(angle)

    if not angles:
        return {
            "available": False,
            "message": "Knee-angle movement data is unavailable.",
            "minimum_knee_angle": None,
            "reached_depth": None,
            "returned_to_standing": None,
            "partial_descents": None,
        }

    minimum_angle = min(angles)

    # These values describe the existing squat criteria.
    reached_depth = minimum_angle <= 90
    returned_to_standing = angles[-1] >= 160

    # Count distinct descent events, avoiding consecutive-frame duplicates.
    partial_descents = 0
    in_descent = False

    for angle in angles:
        if angle < 160 and not in_descent:
            partial_descents += 1
            in_descent = True

        elif angle >= 160:
            in_descent = False

    if repetitions > 0:
        message = (
            f"{repetitions} complete squat repetition(s) were detected "
            "by the existing squat analysis."
        )
    elif not reached_depth:
        message = (
            "The video contains partial descents, but the configured "
            "squat-depth threshold was not reached."
        )
    elif not returned_to_standing:
        message = (
            "The video reached the configured squat-depth threshold, "
            "but a return to standing was not detected."
        )
    else:
        message = (
            "Movement was detected, but no complete repetition was "
            "confirmed by the existing squat analysis."
        )

    return {
        "available": True,
        "message": message,
        "minimum_knee_angle": round(minimum_angle, 2),
        "reached_depth": reached_depth,
        "returned_to_standing": returned_to_standing,
        "partial_descents": partial_descents,
    }


def _video_summary(
    assessment: Dict[str, Any],
) -> Dict[str, Any]:

    return {
        "repetitions": assessment.get("repetitions", 0),
        "assessment_status": assessment.get(
            "assessment_status",
            "Unknown",
        ),
        "performance_score": assessment.get("performance_score"),
    }


def build_squat_comparison_report(
    assessment_a: Dict[str, Any],
    assessment_b: Dict[str, Any],
    curve_a: Optional[List[Dict[str, Any]]] = None,
    curve_b: Optional[List[Dict[str, Any]]] = None,
    comparison_mode: str = "progress",
) -> Dict[str, Any]:
    """
    Build an athlete-facing Squat comparison report.

    This module does not run pose detection, change squat thresholds,
    calculate repetitions, or modify existing squat assessments.
    """

    assessment_a = assessment_a or {}
    assessment_b = assessment_b or {}

    metrics_a = _get_metrics(assessment_a)
    metrics_b = _get_metrics(assessment_b)

    repetitions_a = assessment_a.get("repetitions", 0) or 0
    repetitions_b = assessment_b.get("repetitions", 0) or 0

    status_a = assessment_a.get("assessment_status")
    status_b = assessment_b.get("assessment_status")

    valid_repetitions = repetitions_a > 0 and repetitions_b > 0

    valid_scores = (
        valid_repetitions
        and assessment_a.get("performance_score") is not None
        and assessment_b.get("performance_score") is not None
    )

    metric_comparisons = []

    for key in METRIC_CONFIG:
        comparison = _compare_metric(key, metrics_a, metrics_b)

        if comparison is not None:
            metric_comparisons.append(comparison)

    diagnostics_a = _movement_diagnostics(
        assessment_a,
        curve_a,
    )

    diagnostics_b = _movement_diagnostics(
        assessment_b,
        curve_b,
    )

    observations = [
        f"Video A contains {repetitions_a} detected complete repetitions.",
        f"Video B contains {repetitions_b} detected complete repetitions.",
        f"Video A assessment status: {status_a or 'Unknown'}.",
        f"Video B assessment status: {status_b or 'Unknown'}.",
    ]

    if diagnostics_a["available"]:
        observations.append(
            f"Video A: {diagnostics_a['message']}"
        )

    if diagnostics_b["available"]:
        observations.append(
            f"Video B: {diagnostics_b['message']}"
        )

    limitations = [
        "A larger range of motion is not automatically better; "
        "it must be interpreted in the context of exercise technique.",
        "Smaller left-right symmetry differences indicate closer "
        "measured similarity but do not independently establish "
        "overall performance.",
        "Trunk angle change is descriptive and should not "
        "automatically be classified as better or worse.",
        "Video angle, visibility, camera position and movement "
        "execution can affect pose measurements.",
    ]

    if not valid_repetitions:
        progress_verdict = (
            "Not enough complete repetitions to determine "
            "performance progress."
        )

        limitations.insert(
            0,
            "Repetition-based performance scoring is unavailable "
            "when both videos do not contain complete repetitions.",
        )

    elif not valid_scores:
        progress_verdict = (
            "Complete repetitions were detected, but valid "
            "performance scores are unavailable for comparison."
        )

        limitations.insert(
            0,
            "Performance-score progress cannot be established "
            "without valid scores from both videos.",
        )

    else:
        score_a = float(assessment_a["performance_score"])
        score_b = float(assessment_b["performance_score"])
        score_difference = round(score_b - score_a, 2)

        if score_difference > 0:
            progress_verdict = (
                f"Video B has a {score_difference}-point higher "
                "performance score than Video A."
            )
        elif score_difference < 0:
            progress_verdict = (
                f"Video B has a {abs(score_difference)}-point lower "
                "performance score than Video A."
            )
        else:
            progress_verdict = (
                "Both videos have the same performance score."
            )

    if curve_a and curve_b:
        limitations.append(
            "Knee-angle curves are normalized to full-video duration. "
            "They are not guaranteed to represent corresponding "
            "movement phases or repetitions."
        )
    else:
        limitations.append(
            "Movement-curve comparison is unavailable because "
            "complete curve data was not supplied."
        )

    report_verdict = progress_verdict

    if not valid_repetitions:
        report_verdict += (
            " Review the movement observations and record another "
            "video containing complete repetitions."
        )

    insights = {
        "progress_verdict": progress_verdict,
        "observations": observations,
        "limitations": limitations,
        "repetition_comparison_valid": valid_repetitions,
        "score_comparison_valid": valid_scores,
        "video_a": _video_summary(assessment_a),
        "video_b": _video_summary(assessment_b),
        "movement_diagnostics": {
            "video_a": diagnostics_a,
            "video_b": diagnostics_b,
        },
        "metric_comparisons": metric_comparisons,
        "report_verdict": report_verdict,
        "progress_determined": valid_scores,
    }

    return {
        "activity": "Squat Comparison",
        "comparison_mode": comparison_mode,
        "title": (
            "Squat Progress Comparison"
            if comparison_mode == "progress"
            else "Squat Reference Comparison"
        ),
        "metrics": metric_comparisons,
        "insights": insights,
    }
