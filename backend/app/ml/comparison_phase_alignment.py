
"""
Squat movement-phase alignment.

This module is isolated from the original Squat analysis pipeline.
It detects movement attempts and aligns observed descent, bottom,
and (only when observed) ascent phases.

It does not count repetitions or modify assessment results.
"""

from typing import Any


STANDING_THRESHOLD = 160.0
DEPTH_THRESHOLD = 90.0
OUTPUT_POINTS = 101


def _get_angle(point: dict) -> float | None:
    try:
        value = float(point.get("knee_angle"))

        if 0 < value <= 180:
            return value

    except (TypeError, ValueError, AttributeError):
        pass

    return None


def _clean_curve(curve: list[dict]) -> list[float]:
    if not isinstance(curve, list):
        return []

    values = [_get_angle(point) for point in curve]
    valid = [value for value in values if value is not None]

    if len(valid) < 5:
        return []

    cleaned = []
    last_value = valid[0]

    for value in values:
        if value is not None:
            last_value = value

        cleaned.append(last_value)

    return cleaned


def _smooth(values: list[float], window: int = 3) -> list[float]:
    if len(values) < window:
        return values[:]

    result = []

    for index in range(len(values)):
        start = max(0, index - window // 2)
        end = min(len(values), index + window // 2 + 1)

        result.append(
            sum(values[start:end]) / (end - start)
        )

    return result


def _find_attempts(
    values: list[float],
    standing_threshold: float = STANDING_THRESHOLD,
    min_drop: float = 12.0
) -> list[dict[str, Any]]:
    """
    Detect movement attempts, not confirmed repetitions.
    """

    attempts = []
    smoothed = _smooth(values)

    index = 0
    total = len(smoothed)

    while index < total:

        if smoothed[index] >= standing_threshold:
            index += 1
            continue

        start = index
        minimum_index = start
        minimum_angle = smoothed[start]

        end = total - 1
        returned = False

        cursor = start + 1

        while cursor < total:

            if smoothed[cursor] < minimum_angle:
                minimum_angle = smoothed[cursor]
                minimum_index = cursor

            if smoothed[cursor] >= standing_threshold:
                end = cursor
                returned = True
                break

            cursor += 1

        start_index = max(0, start - 1)
        start_angle = smoothed[start_index]

        if start_angle - minimum_angle >= min_drop:
            attempts.append({
                "start": start_index,
                "bottom": minimum_index,
                "end": end,
                "minimum_angle": round(minimum_angle, 2),
                "reached_depth": minimum_angle <= DEPTH_THRESHOLD,
                "returned_to_standing": returned,
                "ascent_observed": returned
            })

        index = end + 1 if returned else total

    return attempts


def _interpolate_phase(
    values: list[float],
    start: int,
    end: int,
    output_points: int
) -> list[float]:
    """Resample an observed movement section."""

    if not values:
        return []

    start = max(0, min(start, len(values) - 1))
    end = max(start, min(end, len(values) - 1))

    section = values[start:end + 1]

    if len(section) == 1:
        return [
            round(section[0], 2)
            for _ in range(output_points)
        ]

    result = []

    for index in range(output_points):

        position = (
            index * (len(section) - 1)
            / (output_points - 1)
        )

        left = int(position)
        right = min(left + 1, len(section) - 1)
        fraction = position - left

        interpolated = (
            section[left] * (1 - fraction)
            + section[right] * fraction
        )

        result.append(round(interpolated, 2))

    return result


def _align_attempt(
    values: list[float],
    attempt: dict
) -> dict:
    """
    Align descent and bottom.

    An ascent is included only if the video actually returns
    to the standing threshold.
    """

    start = attempt["start"]
    bottom = attempt["bottom"]
    end = attempt["end"]

    descent = _interpolate_phase(
        values,
        start,
        bottom,
        41
    )

    bottom_phase = _interpolate_phase(
        values,
        bottom,
        bottom,
        11
    )

    aligned = descent[:40] + bottom_phase

    if attempt["ascent_observed"]:

        ascent = _interpolate_phase(
            values,
            bottom,
            end,
            51
        )

        aligned.extend(ascent[1:])

    else:
        # Do not manufacture an ascent for an incomplete attempt.
        aligned.extend([None] * 50)

    return {
        "curve": aligned,
        "minimum_angle": attempt["minimum_angle"],
        "reached_depth": attempt["reached_depth"],
        "returned_to_standing": attempt["returned_to_standing"],
        "ascent_observed": attempt["ascent_observed"],
        "phases": {
            "descent": [0, 39],
            "bottom": [40, 50],
            "ascent": (
                [51, 100]
                if attempt["ascent_observed"]
                else None
            )
        }
    }


def align_squat_curves(
    curve_a: list[dict],
    curve_b: list[dict]
) -> dict:
    """
    Align the largest detectable movement attempt in each video.

    Incomplete attempts remain incomplete. Missing ascent data
    is represented as None rather than fabricated values.
    """

    values_a = _clean_curve(curve_a)
    values_b = _clean_curve(curve_b)

    if not values_a or not values_b:
        return {
            "available": False,
            "message": (
                "Insufficient valid knee-angle data "
                "for phase alignment."
            ),
            "video_a": None,
            "video_b": None,
            "aligned_curve": []
        }

    attempts_a = _find_attempts(values_a)
    attempts_b = _find_attempts(values_b)

    if not attempts_a or not attempts_b:
        return {
            "available": False,
            "message": (
                "A detectable movement attempt was not found "
                "in both videos. Full-video curves remain "
                "descriptive only."
            ),
            "video_a": None,
            "video_b": None,
            "aligned_curve": []
        }

    selected_a = max(
        attempts_a,
        key=lambda item: (
            values_a[item["start"]] - item["minimum_angle"]
        )
    )

    selected_b = max(
        attempts_b,
        key=lambda item: (
            values_b[item["start"]] - item["minimum_angle"]
        )
    )

    aligned_a = _align_attempt(values_a, selected_a)
    aligned_b = _align_attempt(values_b, selected_b)

    comparison_curve = []

    for index in range(OUTPUT_POINTS):

        angle_a = aligned_a["curve"][index]
        angle_b = aligned_b["curve"][index]

        difference = (
            round(angle_b - angle_a, 2)
            if angle_a is not None and angle_b is not None
            else None
        )

        comparison_curve.append({
            "phase_percent": index,
            "knee_angle_a": angle_a,
            "knee_angle_b": angle_b,
            "difference": difference
        })

    return {
        "available": True,
        "alignment_method": "descent_bottom_ascent",
        "message": (
            "Movement attempts were phase-aligned. "
            "Unobserved ascent phases are left unavailable "
            "and are not treated as completed repetitions."
        ),
        "video_a": aligned_a,
        "video_b": aligned_b,
        "aligned_curve": comparison_curve,
        "attempt_counts": {
            "video_a": len(attempts_a),
            "video_b": len(attempts_b)
        },
        "complete_attempts": {
            "video_a": sum(
                item["reached_depth"]
                and item["returned_to_standing"]
                for item in attempts_a
            ),
            "video_b": sum(
                item["reached_depth"]
                and item["returned_to_standing"]
                for item in attempts_b
            )
        }
    }
