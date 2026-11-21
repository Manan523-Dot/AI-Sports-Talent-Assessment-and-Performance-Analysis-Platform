
from typing import Any


# Metrics that are useful for athlete-facing comparison.
# Technical detection/coverage counters are deliberately excluded.
ALLOWED_METRICS = {
    "performance_score",
    "repetitions",
    "knee_range_of_motion",
    "hip_range_of_motion",
    "knee_symmetry_difference",
    "hip_symmetry_difference",
    "trunk_angle_change",
    "average_bottom_angle",
    "deepest_rep_angle",
    "descent_time",
    "ascent_time",
    "average_rep_time",
}


def _get_value(data: dict, path: str):
    current = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)

    return current


def _round(value):
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return value


def _metric_name(path: str) -> str:
    name = path.split(".")[-1]

    names = {
        "performance_score": "Performance Score",
        "repetitions": "Completed Repetitions",
        "knee_range_of_motion": "Knee Range of Motion",
        "hip_range_of_motion": "Hip Range of Motion",
        "knee_symmetry_difference": "Knee Symmetry Difference",
        "hip_symmetry_difference": "Hip Symmetry Difference",
        "trunk_angle_change": "Trunk Angle Change",
        "average_bottom_angle": "Average Squat Bottom Angle",
        "deepest_rep_angle": "Deepest Squat Angle",
        "descent_time": "Average Descent Time",
        "ascent_time": "Average Ascent Time",
        "average_rep_time": "Average Repetition Time",
    }

    return names.get(name, name.replace("_", " ").title())


def _interpret_metric(name: str, a, b) -> str:
    if a is None or b is None:
        return "Not enough valid data to compare this metric."

    difference = round(float(b) - float(a), 2)

    if abs(difference) < 0.01:
        return "No meaningful change detected."

    if name in {
        "knee_symmetry_difference",
        "hip_symmetry_difference",
    }:
        if difference < 0:
            return (
                f"Side-to-side difference decreased by "
                f"{abs(difference):.2f} degrees."
            )
        return (
            f"Side-to-side difference increased by "
            f"{difference:.2f} degrees."
        )

    if name == "performance_score":
        direction = "increased" if difference > 0 else "decreased"
        return f"Performance score {direction} by {abs(difference):.2f}."

    if name == "repetitions":
        direction = "increased" if difference > 0 else "decreased"
        return f"Completed repetitions {direction} by {abs(difference):.0f}."

    if name in {
        "knee_range_of_motion",
        "hip_range_of_motion",
    }:
        direction = "increased" if difference > 0 else "decreased"
        return (
            f"Range of motion {direction} by "
            f"{abs(difference):.2f} degrees. "
            "A change alone does not establish better technique."
        )

    if name == "trunk_angle_change":
        direction = "increased" if difference > 0 else "decreased"
        return (
            f"Trunk angle change {direction} by "
            f"{abs(difference):.2f} degrees. "
            "Interpret alongside video posture and movement quality."
        )

    if name in {
        "average_bottom_angle",
        "deepest_rep_angle",
    }:
        return (
            f"Measured angle changed by {difference:.2f} degrees. "
            "Review the video to assess squat depth."
        )

    if name in {
        "descent_time",
        "ascent_time",
        "average_rep_time",
    }:
        return (
            f"Movement time changed by {difference:.2f} seconds. "
            "Faster movement is not automatically better."
        )

    return "Metric changed; review the movement context."


def _movement_observations(
    assessment_a: dict,
    assessment_b: dict,
    curve: list,
) -> list:
    observations = []

    reps_a = assessment_a.get("repetitions", 0) or 0
    reps_b = assessment_b.get("repetitions", 0) or 0

    if reps_a == 0 or reps_b == 0:
        observations.append(
            "At least one video has no complete squat repetitions. "
            "Performance scores are therefore unavailable."
        )

    if curve:
        angles_a = [
            point.get("knee_angle_a")
            for point in curve
            if isinstance(point.get("knee_angle_a"), (int, float))
        ]
        angles_b = [
            point.get("knee_angle_b")
            for point in curve
            if isinstance(point.get("knee_angle_b"), (int, float))
        ]

        if angles_a and angles_b:
            minimum_a = min(angles_a)
            minimum_b = min(angles_b)

            observations.append(
                f"Lowest observed knee angle: first video "
                f"{minimum_a:.2f} degrees; second video "
                f"{minimum_b:.2f} degrees."
            )

            if minimum_b < minimum_a:
                observations.append(
                    "The second video reached a lower knee angle, "
                    "indicating a greater measured knee bend."
                )
            elif minimum_b > minimum_a:
                observations.append(
                    "The first video reached a lower knee angle, "
                    "indicating a greater measured knee bend."
                )
            else:
                observations.append(
                    "Both videos reached a similar minimum knee angle."
                )

    if reps_a == 0 or reps_b == 0:
        observations.append(
            "Partial movements can still be reviewed, but they are "
            "not counted as completed repetitions."
        )

    return observations


def enhance_squat_comparison(
    comparison: dict,
    assessment_a: dict,
    assessment_b: dict,
) -> dict:
    """
    Improve an existing squat comparison result.

    This function does not run pose detection, change squat thresholds,
    calculate new performance scores, or modify assessment records.
    It only improves the comparison report.
    """

    comparison = dict(comparison)

    original_metrics = comparison.get("metrics", [])
    filtered_metrics = []

    for item in original_metrics:
        path = item.get("metric", "")
        metric = path.split(".")[-1]

        if metric not in ALLOWED_METRICS:
            continue

        value_a = item.get("value_a")
        value_b = item.get("value_b")

        if value_a is None or value_b is None:
            continue

        filtered_metrics.append({
            "metric": path,
            "metric_name": _metric_name(path),
            "value_a": _round(value_a),
            "value_b": _round(value_b),
            "difference": _round(
                float(value_b) - float(value_a)
            ),
            "interpretation": _interpret_metric(
                metric,
                value_a,
                value_b,
            ),
        })

    curve = comparison.get("curve_comparison", [])

    comparison["metrics"] = filtered_metrics

    comparison["observations"] = _movement_observations(
        assessment_a,
        assessment_b,
        curve,
    )

    comparison["limitations"] = [
        "Performance scores are unavailable when complete repetitions "
        "cannot be detected.",
        "Partial movements are described using measured movement data "
        "and are not treated as completed repetitions.",
        "Camera position, visibility, and video quality can affect "
        "angle measurements.",
        "A change in range of motion does not automatically indicate "
        "better technique.",
        "This comparison is an analysis aid, not a replacement for "
        "professional coaching.",
    ]

    return comparison
