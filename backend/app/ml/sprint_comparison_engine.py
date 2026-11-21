
import math
import statistics


def _safe_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _get_nested(data, *keys):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def _extract_knee_angles(features):
    """
    Extract knee-angle values from the existing Sprint feature JSON.
    Supports numeric lists and lists of dictionaries.
    """

    data = features.get("knee_angle_data", [])

    if not isinstance(data, list):
        return []

    angles = []

    for item in data:
        value = None

        if isinstance(item, (int, float)):
            value = item

        elif isinstance(item, dict):
            for key in (
                "knee_angle",
                "average_knee_angle",
                "angle",
                "left_knee_angle",
                "right_knee_angle"
            ):
                value = item.get(key)

                if value is not None:
                    break

        value = _safe_float(value)

        if value is not None:
            angles.append(value)

    return angles


def _normalize_curve(values, points=101):
    """
    Normalize a movement curve to 0–100% of recording duration.
    This is recording-time alignment, not stride-phase alignment.
    """

    if not values:
        return []

    if len(values) == 1:
        return [round(values[0], 2)] * points

    normalized = []

    for index in range(points):
        position = (index / (points - 1)) * (len(values) - 1)

        left_index = int(position)
        right_index = min(left_index + 1, len(values) - 1)

        fraction = position - left_index

        value = (
            values[left_index] * (1 - fraction)
            + values[right_index] * fraction
        )

        normalized.append(round(value, 2))

    return normalized


def _movement_value(features, movement, metric):
    value = _get_nested(features, movement, metric)

    return _safe_float(value)


def _assessment_score(assessment):
    for key in (
        "overall_score",
        "performance_score",
        "score"
    ):
        value = _safe_float(assessment.get(key))

        if value is not None:
            return value

    return None


def _metric_row(
    name,
    value_a,
    value_b,
    unit="",
    interpretation="descriptive"
):
    value_a = _safe_float(value_a)
    value_b = _safe_float(value_b)

    difference = None

    if value_a is not None and value_b is not None:
        difference = round(value_b - value_a, 2)

    return {
        "metric": name,
        "video_a": value_a,
        "video_b": value_b,
        "difference": difference,
        "unit": unit,
        "interpretation": interpretation
    }


def _build_metric_rows(features_a, assessment_a, features_b, assessment_b):
    rows = []

    # Overall assessment scores
    rows.append(
        _metric_row(
            "Overall Performance Score",
            _assessment_score(assessment_a),
            _assessment_score(assessment_b),
            "points",
            "higher_score"
        )
    )

    # Movement range of motion
    movements = [
        ("Knee Movement ROM", "knee_movement"),
        ("Hip Movement ROM", "hip_movement"),
        ("Trunk Movement ROM", "trunk_movement")
    ]

    for label, movement in movements:
        rows.append(
            _metric_row(
                label,
                _movement_value(features_a, movement, "average_rom"),
                _movement_value(features_b, movement, "average_rom"),
                "degrees",
                "descriptive_rom"
            )
        )

        rows.append(
            _metric_row(
                label + " Symmetry Difference",
                _movement_value(features_a, movement, "symmetry_difference"),
                _movement_value(features_b, movement, "symmetry_difference"),
                "degrees",
                "lower_is_better"
            )
        )

    # Recording duration
    rows.append(
        _metric_row(
            "Recording Duration",
            features_a.get("duration_seconds"),
            features_b.get("duration_seconds"),
            "seconds",
            "descriptive_duration"
        )
    )

    # Frame count
    rows.append(
        _metric_row(
            "Frame Count",
            features_a.get("frame_count"),
            features_b.get("frame_count"),
            "frames",
            "descriptive"
        )
    )

    return [
        row for row in rows
        if row["video_a"] is not None or row["video_b"] is not None
    ]


def _build_observations(metrics, comparison_mode):
    observations = []

    for row in metrics:
        name = row["metric"]
        value_a = row["video_a"]
        value_b = row["video_b"]
        difference = row["difference"]

        if value_a is None or value_b is None:
            observations.append(
                f"{name}: comparison unavailable because one or both videos "
                "do not contain this measurement."
            )
            continue

        if difference is None or abs(difference) < 0.01:
            observations.append(
                f"{name}: no meaningful numerical difference was detected."
            )
            continue

        if row["interpretation"] == "higher_score":
            direction = (
                "increased" if difference > 0 else "decreased"
            )

            observations.append(
                f"{name} {direction} by {abs(difference):.2f} points."
            )

        elif row["interpretation"] == "lower_is_better":
            direction = (
                "decreased" if difference < 0 else "increased"
            )

            observations.append(
                f"{name} {direction} by {abs(difference):.2f} degrees. "
                "A smaller symmetry difference indicates closer left-right "
                "measurements when both sides are reliably visible."
            )

        elif row["interpretation"] == "descriptive_rom":
            direction = (
                "increased" if difference > 0 else "decreased"
            )

            observations.append(
                f"{name} {direction} by {abs(difference):.2f} degrees. "
                "This is a descriptive change and does not automatically "
                "indicate better sprint technique."
            )

        elif row["interpretation"] == "descriptive_duration":
            observations.append(
                f"{name} changed by {difference:.2f} seconds. "
                "Recording duration is not a sprint speed measurement."
            )

        else:
            observations.append(
                f"{name} changed by {difference:.2f}."
            )

    if comparison_mode == "progress":
        observations.insert(
            0,
            "Progress comparison: Video A is treated as the earlier "
            "recording and Video B as the newer recording."
        )
    else:
        observations.insert(
            0,
            "Reference comparison: Video A is treated as the athlete "
            "recording and Video B as the reference recording."
        )

    return observations


def build_sprint_comparison(
    features_a,
    assessment_a,
    features_b,
    assessment_b,
    comparison_mode="progress"
):
    """
    Build a comparison using the existing Sprint feature and assessment
    outputs. Does not modify the original Sprint analysis pipeline.
    """

    if comparison_mode not in ("progress", "reference"):
        raise ValueError(
            "comparison_mode must be 'progress' or 'reference'."
        )

    knee_a = _extract_knee_angles(features_a)
    knee_b = _extract_knee_angles(features_b)

    normalized_a = _normalize_curve(knee_a)
    normalized_b = _normalize_curve(knee_b)

    aligned_curve = []

    if normalized_a and normalized_b:
        for index in range(101):
            aligned_curve.append({
                "phase_percent": index,
                "knee_angle_a": normalized_a[index],
                "knee_angle_b": normalized_b[index]
            })

    metrics = _build_metric_rows(
        features_a,
        assessment_a,
        features_b,
        assessment_b
    )

    observations = _build_observations(
        metrics,
        comparison_mode
    )

    limitations = [
        "The comparison uses the measurements produced by the existing "
        "Sprint analysis pipeline.",
        "The knee-angle curves are aligned by normalized recording time, "
        "not by individual stride or gait phase.",
        "The current Sprint pipeline does not directly measure running "
        "speed, stride length, ground contact time or validated stride "
        "consistency.",
        "Differences in camera position, distance, frame rate, visibility "
        "and recording conditions can affect comparisons.",
        "A higher range of motion is not automatically better.",
        "The assessment is a prototype analysis and should not be treated "
        "as a scientifically validated athletic ranking."
    ]

    return {
        "activity": "Sprint Comparison",
        "comparison_mode": comparison_mode,

        "video_a": {
            "assessment": assessment_a,
            "knee_angle_count": len(knee_a)
        },

        "video_b": {
            "assessment": assessment_b,
            "knee_angle_count": len(knee_b)
        },

        "metrics": metrics,

        "phase_alignment": {
            "method": "normalized_recording_time",
            "aligned_curve": aligned_curve
        },

        "observations": observations,
        "limitations": limitations
    }
