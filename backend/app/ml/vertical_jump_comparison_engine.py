
"""Vertical Jump comparison helpers.

Uses the existing Vertical Jump analysis outputs.
Does not modify the single-video analysis or assessment pipeline.
"""

import numpy as np


def _number(value):
    try:
        number = float(value)
        return round(number, 2) if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _metric_rows(assessment_a, assessment_b, features_a, features_b):
    metrics_a = (assessment_a or {}).get("metrics") or {}
    metrics_b = (assessment_b or {}).get("metrics") or {}

    rows = []

    definitions = [
        (
            "estimated_jump_height_cm",
            "Estimated jump height",
            "cm",
            "Physics-based estimate derived from candidate flight time; it is not a direct jump-height measurement.",
        ),
        (
            "flight_time",
            "Candidate flight time",
            "s",
            "Estimated from knee-angle movement events. Feet-off and feet-on are not directly confirmed.",
        ),
        (
            "knee_range_of_motion",
            "Overall knee range of motion",
            "°",
            "Descriptive knee movement range; a larger value is not automatically better.",
        ),
        (
            "knee_rom_difference",
            "Left-right knee ROM difference",
            "°",
            "A smaller difference indicates closer measured left-right ROM values. Camera angle and landmark visibility can affect this measurement.",
        ),
        (
            "takeoff_knee_angle",
            "Takeoff knee angle",
            "°",
            "Descriptive angle at the detected candidate takeoff event.",
        ),
        (
            "landing_knee_angle",
            "Landing knee angle",
            "°",
            "Descriptive angle at the detected candidate landing event.",
        ),
        (
            "bottom_knee_angle",
            "Bottom knee angle",
            "°",
            "Descriptive knee angle at the detected deepest bend.",
        ),
    ]

    for key, label, unit, interpretation in definitions:
        a = _number(metrics_a.get(key, features_a.get(key)))
        b = _number(metrics_b.get(key, features_b.get(key)))

        if a is None and b is None:
            continue

        rows.append({
            "metric": key,
            "label": label,
            "video_a": a,
            "video_b": b,
            "difference": (
                _number(b - a)
                if a is not None and b is not None
                else None
            ),
            "unit": unit,
            "interpretation": (
                interpretation
                if a is not None and b is not None
                else "This measurement was not available in both videos."
            ),
        })

    for key, label in (
        ("left_knee_rom", "Left knee ROM"),
        ("right_knee_rom", "Right knee ROM"),
    ):
        a = _number(features_a.get(key))
        b = _number(features_b.get(key))

        if a is None and b is None:
            continue

        rows.append({
            "metric": key,
            "label": label,
            "video_a": a,
            "video_b": b,
            "difference": (
                _number(b - a)
                if a is not None and b is not None
                else None
            ),
            "unit": "°",
            "interpretation": (
                "Descriptive side-specific knee range of motion."
                if a is not None and b is not None
                else "This measurement was not available in both videos."
            ),
        })

    score_a = _number(
        (assessment_a or {}).get("performance_score")
    )
    score_b = _number(
        (assessment_b or {}).get("performance_score")
    )

    if score_a is not None or score_b is not None:
        method_a = (assessment_a or {}).get("assessment_method")
        method_b = (assessment_b or {}).get("assessment_method")

        same_method = method_a == method_b

        rows.insert(0, {
            "metric": "performance_score",
            "label": "Assessment score",
            "video_a": score_a,
            "video_b": score_b,
            "difference": (
                _number(score_b - score_a)
                if score_a is not None and score_b is not None
                else None
            ),
            "unit": "/100",
            "interpretation": (
                "Both scores use the same assessment method. "
                "Treat them as experimental outputs, not validated athletic ratings."
                if same_method
                else "Assessment methods differ or are unavailable; score comparison may not be meaningful."
            ),
        })

    return rows


def _normalized_curve(features, points=101):
    data = (features or {}).get("knee_angle_data") or []

    valid = []

    for item in data:
        # Preserve the original timestamp precision.
        # Do not use _number() here because it rounds values to 2 decimals.
        try:
            time_value = float(item.get("time"))
            angle_value = float(item.get("knee_angle"))
        except (TypeError, ValueError, AttributeError):
            continue

        if not np.isfinite(time_value) or not np.isfinite(angle_value):
            continue

        if not 0 <= angle_value <= 180:
            continue

        valid.append((time_value, angle_value))

    if len(valid) < 5:
        return []

    valid.sort(key=lambda item: item[0])

    times = np.array(
        [item[0] for item in valid],
        dtype=float
    )

    angles = np.array(
        [item[1] for item in valid],
        dtype=float
    )

    # Remove duplicate timestamps before interpolation.
    unique_times, unique_indices = np.unique(
        times,
        return_index=True
    )

    angles = angles[unique_indices]
    times = unique_times

    if len(times) < 5 or times[-1] <= times[0]:
        return []

    # Smooth the curve with a small moving median.
    smoothed = []

    for index in range(len(angles)):
        start = max(0, index - 2)
        end = min(len(angles), index + 3)

        smoothed.append(
            float(np.median(angles[start:end]))
        )

    smoothed = np.array(smoothed, dtype=float)

    normalized_progress = np.linspace(0, 100, points)

    source_progress = (
        (times - times[0])
        / (times[-1] - times[0])
        * 100
    )

    interpolated = np.interp(
        normalized_progress,
        source_progress,
        smoothed
    )

    return [
        {
            "time_percent": int(progress),
            "knee_angle": round(float(angle), 2),
        }
        for progress, angle in zip(
            normalized_progress,
            interpolated
        )
    ]


def build_vertical_jump_comparison(
    assessment_a,
    assessment_b,
    features_a,
    features_b,
    mode,
):
    curve_a = _normalized_curve(features_a)
    curve_b = _normalized_curve(features_b)

    aligned_curve = []

    if curve_a and curve_b:
        for point_a, point_b in zip(curve_a, curve_b):
            aligned_curve.append({
                "progress": point_a["time_percent"],
                "video_a": point_a["knee_angle"],
                "video_b": point_b["knee_angle"],
                "difference": round(
                    point_b["knee_angle"]
                    - point_a["knee_angle"],
                    2
                ),
            })

    metrics = _metric_rows(
        assessment_a,
        assessment_b,
        features_a,
        features_b,
    )

    detected_a = bool(
        features_a.get("jump_detected", False)
    )
    detected_b = bool(
        features_b.get("jump_detected", False)
    )

    observations = [
        "Both videos were processed independently using the existing Vertical Jump pose-detection, movement-analysis and assessment functions.",
        "Jump height is estimated from candidate flight time using a physics-based formula; it is not directly measured.",
        "The knee-angle curves are aligned by normalized recording time (0–100%), not by actual jump phases.",
    ]

    if detected_a and detected_b:
        observations.append(
            "A candidate crouch-extension-flexion movement cycle was detected in both videos."
        )
    elif detected_a or detected_b:
        observations.append(
            "A candidate movement cycle was detected in only one video. Interpret the metric comparison with caution."
        )
    else:
        observations.append(
            "A complete candidate jump cycle was not detected in either video. Jump height and flight-time comparisons are not reliable."
        )

    if not aligned_curve:
        observations.append(
            "A comparison curve could not be generated because one or both videos had insufficient valid knee-angle measurements."
        )

    limitations = [
        "The current detection uses knee-angle patterns. It cannot directly confirm that the feet leave or contact the ground.",
        "Estimated jump height is calculated from candidate flight time and assumes symmetrical flight.",
        "Camera angle, distance, frame rate, occlusion and landmark visibility can affect the measurements.",
        "Knee range of motion and takeoff/landing angles are descriptive measurements; higher or lower values are not automatically better.",
        "Normalized recording-time curves do not align the actual crouch, takeoff, flight and landing phases.",
        "Progress comparison is most meaningful when both recordings use similar camera position, distance, framing and recording conditions.",
        "Assessment scores are experimental outputs and should not be treated as scientifically validated athletic ratings.",
    ]

    title = (
        "Vertical Jump Reference Comparison"
        if mode == "reference"
        else "Vertical Jump Progress Comparison"
    )

    return {
        "activity": "Vertical Jump",
        "exercise": "Vertical Jump",
        "comparison_mode": mode,
        "title": title,
        "metrics": metrics,
        "phase_alignment": {
            "available": bool(aligned_curve),
            "alignment_method": (
                "Knee-angle curve resampled to normalized recording time (0–100%)"
            ),
            "aligned_curve": aligned_curve,
            "message": (
                "This curve compares normalized recording time, not actual jump phases."
                if aligned_curve
                else "Insufficient valid knee-angle data to create a comparison curve."
            ),
        },
        "curve_comparison": aligned_curve,
        "observations": observations,
        "limitations": limitations,
    }
