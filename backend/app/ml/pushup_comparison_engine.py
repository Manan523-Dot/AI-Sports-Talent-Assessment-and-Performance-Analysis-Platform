
import numpy as np
import pandas as pd


# Metrics where a lower value generally indicates better consistency/control.
LOWER_IS_BETTER = {
    "body_alignment_deviation",
    "elbow_symmetry_difference",
    "rep_time_difference",
}

# Metrics where a higher value is generally favourable.
HIGHER_IS_BETTER = {
    "performance_score",
}

# These metrics should be reported as changes, not automatically ranked.
CONTEXT_DEPENDENT = {
    "elbow_range_of_motion",
    "average_bottom_elbow_angle",
    "average_rep_time",
    "repetitions",
}


def _number(value):
    try:
        if value is None:
            return None

        result = float(value)
        return result if np.isfinite(result) else None

    except (TypeError, ValueError):
        return None


def _get_metrics(assessment):
    metrics = assessment.get("metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def _get_score(assessment):
    return _number(assessment.get("performance_score"))


def _get_coverage(assessment):
    return _number(assessment.get("score_coverage_percent"))


def _metric_change(name, value_a, value_b):
    a = _number(value_a)
    b = _number(value_b)

    if a is None or b is None:
        return {
            "metric": name,
            "video_a": a,
            "video_b": b,
            "difference": None,
            "change_percent": None,
            "direction": "unavailable",
            "interpretation": (
                "This metric could not be compared because one or both "
                "videos did not provide a reliable value."
            ),
        }

    difference = b - a

    change_percent = (
        difference / abs(a) * 100
        if abs(a) > 1e-9
        else None
    )

    if abs(difference) < 1e-6:
        direction = "unchanged"
        interpretation = "No meaningful change was detected."

    elif name in LOWER_IS_BETTER:
        direction = "improved" if difference < 0 else "declined"

        interpretation = (
            "The value decreased, which is generally favourable for this metric."
            if difference < 0
            else "The value increased; review this measurement alongside the video."
        )

    elif name in HIGHER_IS_BETTER:
        direction = "improved" if difference > 0 else "declined"

        interpretation = (
            "The value increased, which is generally favourable for this metric."
            if difference > 0
            else "The value decreased compared with the earlier attempt."
        )

    elif name in CONTEXT_DEPENDENT:
        direction = "changed"
        interpretation = (
            "The value changed. Its significance depends on movement quality, "
            "exercise technique, and the athlete's intended tempo."
        )

    else:
        direction = "changed"
        interpretation = (
            "The metric changed; review it alongside the movement video."
        )

    return {
        "metric": name,
        "video_a": round(a, 3),
        "video_b": round(b, 3),
        "difference": round(difference, 3),
        "change_percent": (
            round(change_percent, 2)
            if change_percent is not None
            else None
        ),
        "direction": direction,
        "interpretation": interpretation,
    }


def _read_pose_csv(csv_path):
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return None

    required = {"time_seconds"}

    if frame.empty or not required.issubset(frame.columns):
        return None

    frame["time_seconds"] = pd.to_numeric(
        frame["time_seconds"],
        errors="coerce",
    )

    angle_columns = [
        column
        for column in ("left_elbow_angle", "right_elbow_angle")
        if column in frame.columns
    ]

    if not angle_columns:
        return None

    for column in angle_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["average_elbow_angle"] = frame[angle_columns].mean(axis=1)

    frame = frame.dropna(
        subset=["time_seconds", "average_elbow_angle"]
    )

    frame = frame.sort_values("time_seconds")
    frame = frame.drop_duplicates(subset=["time_seconds"])

    if len(frame) < 2:
        return None

    return frame


def _interpolate_rep(frame, start_time, end_time, points=101):
    """
    Extract one repetition and normalize its timeline to 0–100%.
    """
    start_time = _number(start_time)
    end_time = _number(end_time)

    if start_time is None or end_time is None:
        return None

    if end_time <= start_time:
        return None

    rep_frame = frame[
        (frame["time_seconds"] >= start_time)
        & (frame["time_seconds"] <= end_time)
    ]

    if len(rep_frame) < 2:
        return None

    times = rep_frame["time_seconds"].to_numpy(dtype=float)
    angles = rep_frame["average_elbow_angle"].to_numpy(dtype=float)

    duration = times[-1] - times[0]

    if duration <= 0:
        return None

    normalized_times = (times - times[0]) / duration

    target_times = np.linspace(0, 1, points)

    interpolated = np.interp(
        target_times,
        normalized_times,
        angles,
    )

    return interpolated


def _build_rep_curve(frame, assessment):
    """
    Align repetitions individually using detected rep start/end times.

    The final curve represents the average normalized repetition.
    """
    rep_details = assessment.get("rep_details") or []

    if not isinstance(rep_details, list) or not rep_details:
        return None

    rep_curves = []

    for rep in rep_details:
        if not isinstance(rep, dict):
            continue

        curve = _interpolate_rep(
            frame,
            rep.get("start_time"),
            rep.get("end_time"),
        )

        if curve is not None:
            rep_curves.append(curve)

    if not rep_curves:
        return None

    average_curve = np.mean(
        np.vstack(rep_curves),
        axis=0,
    )

    return {
        "curve": average_curve,
        "repetitions_used": len(rep_curves),
        "repetitions_detected": len(rep_details),
    }


def _build_movement_curve(frame_a, frame_b, assessment_a, assessment_b):
    if frame_a is None or frame_b is None:
        return {
            "available": False,
            "label": "Average elbow angle",
            "unit": "degrees",
            "video_a": [],
            "video_b": [],
            "mean_absolute_difference": None,
            "explanation": (
                "Movement curves could not be generated from both pose CSV files."
            ),
        }

    rep_a = _build_rep_curve(frame_a, assessment_a)
    rep_b = _build_rep_curve(frame_b, assessment_b)

    if rep_a is None or rep_b is None:
        return {
            "available": False,
            "label": "Average elbow angle",
            "unit": "degrees",
            "video_a": [],
            "video_b": [],
            "mean_absolute_difference": None,
            "explanation": (
                "Individual repetitions could not be aligned. "
                "Check that both assessments contain valid repetition start and end times."
            ),
        }

    curve_a = rep_a["curve"]
    curve_b = rep_b["curve"]

    mean_difference = float(
        np.mean(np.abs(curve_b - curve_a))
    )

    phase_points = np.linspace(0, 100, len(curve_a))

    formatted_a = [
        {
            "phase_percent": round(float(phase), 1),
            "elbow_angle": round(float(angle), 2),
        }
        for phase, angle in zip(phase_points, curve_a)
    ]

    formatted_b = [
        {
            "phase_percent": round(float(phase), 1),
            "elbow_angle": round(float(angle), 2),
        }
        for phase, angle in zip(phase_points, curve_b)
    ]

    return {
        "available": True,
        "label": "Average elbow angle",
        "unit": "degrees",
        "video_a": formatted_a,
        "video_b": formatted_b,
        "mean_absolute_difference": round(mean_difference, 2),
        "repetitions_used": {
            "video_a": rep_a["repetitions_used"],
            "video_b": rep_b["repetitions_used"],
        },
        "explanation": (
            "Each detected repetition is individually normalized from its "
            "start to end time. The curves show the average repetition shape "
            "on a common 0–100% movement timeline. This is a comparison of "
            "average movement patterns, not a frame-by-frame identity match."
        ),
    }


def _build_observations(
    score_comparison,
    repetitions_comparison,
    metric_comparisons,
    coverage_a,
    coverage_b,
):
    observations = []

    # Score change is reported, but qualified when coverage differs.
    if score_comparison["direction"] == "improved":
        if (
            coverage_a is not None
            and coverage_b is not None
            and coverage_a != coverage_b
        ):
            observations.append(
                "Performance Score: The score increased, but measurement "
                "coverage differs between the videos. Treat this as a "
                "provisional comparison rather than a definitive improvement."
            )
        else:
            observations.append(
                "Performance Score: The assessment score increased."
            )

    elif score_comparison["direction"] == "declined":
        observations.append(
            "Performance Score: The assessment score decreased."
        )

    if repetitions_comparison["direction"] == "changed":
        observations.append(
            "Repetitions: The detected repetition count changed. "
            "Repetition count alone does not establish better technique."
        )

    for item in metric_comparisons:
        name = item["metric"].replace("_", " ").title()

        if item["direction"] == "improved":
            observations.append(
                f"{name}: {item['interpretation']}"
            )

        elif item["direction"] == "declined":
            observations.append(
                f"{name}: {item['interpretation']}"
            )

        elif item["direction"] == "changed":
            observations.append(
                f"{name}: {item['interpretation']}"
            )

        elif item["direction"] == "unavailable":
            observations.append(
                f"{name}: A direct comparison was unavailable."
            )

    if coverage_a is not None and coverage_b is not None:
        if coverage_a != coverage_b:
            observations.append(
                f"Assessment coverage differs: Video A has {coverage_a:.1f}% "
                f"coverage and Video B has {coverage_b:.1f}% coverage."
            )

    if not observations:
        observations.append(
            "No meaningful differences were identified in the available metrics."
        )

    return observations


def compare_pushup_results(
    assessment_a,
    assessment_b,
    csv_path_a=None,
    csv_path_b=None,
    comparison_mode="progress",
    filename_a=None,
    filename_b=None,
):
    """
    Compare two Push-up assessments.

    Progress:
        Video A = earlier attempt
        Video B = latest attempt

    Reference:
        Video A = reference video
        Video B = athlete video
    """
    assessment_a = assessment_a or {}
    assessment_b = assessment_b or {}

    metrics_a = _get_metrics(assessment_a)
    metrics_b = _get_metrics(assessment_b)

    coverage_a = _get_coverage(assessment_a)
    coverage_b = _get_coverage(assessment_b)

    metric_names = [
        "elbow_range_of_motion",
        "average_bottom_elbow_angle",
        "body_alignment_deviation",
        "elbow_symmetry_difference",
        "average_rep_time",
        "rep_time_difference",
    ]

    metric_comparisons = [
        _metric_change(
            name,
            metrics_a.get(name),
            metrics_b.get(name),
        )
        for name in metric_names
    ]

    score_comparison = _metric_change(
        "performance_score",
        _get_score(assessment_a),
        _get_score(assessment_b),
    )

    repetitions_comparison = _metric_change(
        "repetitions",
        assessment_a.get("repetitions"),
        assessment_b.get("repetitions"),
    )

    # Explicitly qualify score comparisons when measurement coverage differs.
    if (
        score_comparison["direction"] == "improved"
        and coverage_a is not None
        and coverage_b is not None
        and coverage_a != coverage_b
    ):
        score_comparison["interpretation"] = (
            "The score increased, but assessment coverage differs between "
            "the videos. The score change should be interpreted cautiously."
        )
        score_comparison["coverage_difference"] = round(
            coverage_b - coverage_a,
            2,
        )

    frame_a = _read_pose_csv(csv_path_a) if csv_path_a else None
    frame_b = _read_pose_csv(csv_path_b) if csv_path_b else None

    movement_curve = _build_movement_curve(
        frame_a,
        frame_b,
        assessment_a,
        assessment_b,
    )

    observations = _build_observations(
        score_comparison,
        repetitions_comparison,
        metric_comparisons,
        coverage_a,
        coverage_b,
    )

    limitations = [
        "The comparison is based on measurements available from the uploaded videos.",
        "Camera angle, visibility, lighting, and pose-estimation quality can affect measured angles.",
        "A higher repetition count or score does not by itself establish better exercise technique.",
        "The scoring system is an experimental project assessment, not a clinical or certified fitness evaluation.",
        "Elbow range of motion, bottom elbow angle, and repetition duration are reported as changes rather than automatically ranked.",
        "The movement curve compares average normalized repetitions and may hide differences between individual repetitions.",
    ]

    if coverage_a is not None and coverage_b is not None:
        if coverage_a != coverage_b:
            limitations.append(
                "Assessment coverage differs between the two videos. "
                "Performance scores should therefore be interpreted cautiously."
            )

    if (
        metrics_a.get("elbow_symmetry_difference") is None
        or metrics_b.get("elbow_symmetry_difference") is None
    ):
        limitations.append(
            "Elbow symmetry was not available for both videos, so symmetry "
            "cannot be compared directly."
        )

    if not movement_curve["available"]:
        limitations.append(
            "A normalized repetition curve could not be generated for both videos."
        )

    return {
        "exercise": "Push-up",
        "comparison_mode": comparison_mode,
        "video_a": {
            "label": (
                "Earlier attempt"
                if comparison_mode == "progress"
                else "Reference"
            ),
            "filename": filename_a,
            "performance_score": _get_score(assessment_a),
            "repetitions": assessment_a.get("repetitions"),
            "assessment_status": assessment_a.get("assessment_status"),
            "score_coverage_percent": coverage_a,
        },
        "video_b": {
            "label": (
                "Latest attempt"
                if comparison_mode == "progress"
                else "Athlete video"
            ),
            "filename": filename_b,
            "performance_score": _get_score(assessment_b),
            "repetitions": assessment_b.get("repetitions"),
            "assessment_status": assessment_b.get("assessment_status"),
            "score_coverage_percent": coverage_b,
        },
        "score_comparison": score_comparison,
        "repetitions_comparison": repetitions_comparison,
        "metric_comparisons": metric_comparisons,
        "movement_curve_comparison": movement_curve,
        "observations": observations,
        "limitations": limitations,
    }


# Compatibility alias for comparison API conventions.
compare_analyses = compare_pushup_results
