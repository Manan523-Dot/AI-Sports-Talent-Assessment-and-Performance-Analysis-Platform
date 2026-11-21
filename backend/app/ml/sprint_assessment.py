
WEIGHTS = {
    "knee_movement": 0.25,
    "knee_symmetry": 0.20,
    "hip_movement": 0.20,
    "hip_symmetry": 0.15,
    "trunk_movement": 0.10,
}

def find_sprint_metrics(value):
    """
    Find the actual sprint_metrics dictionary, even if the API
    passes it inside another result dictionary or as another argument.
    """
    if not isinstance(value, dict):
        return None

    required_keys = {
        "knee_movement",
        "hip_movement",
        "trunk_movement",
    }

    if required_keys.issubset(value.keys()):
        return value

    for nested_value in value.values():
        if isinstance(nested_value, dict):
            result = find_sprint_metrics(nested_value)
            if result is not None:
                return result

    return None


def get_number(value):
    try:
        if value is None:
            return None

        number = float(value)

        if number != number:  # NaN check
            return None

        return number
    except (TypeError, ValueError):
        return None


def get_average_rom(movement):
    if not isinstance(movement, dict):
        return None

    if movement.get("available") is False:
        return None

    value = get_number(movement.get("average_rom"))

    if value is not None:
        return value

    left = get_number(movement.get("left_rom"))
    right = get_number(movement.get("right_rom"))

    if left is not None and right is not None:
        return (left + right) / 2

    return left if left is not None else right


def get_symmetry_difference(movement):
    if not isinstance(movement, dict):
        return None

    if movement.get("symmetry_available") is False:
        return None

    return get_number(movement.get("symmetry_difference"))


def score_rom(rom):
    """
    Experimental prototype scoring.
    These ranges are not validated professional sprint standards.
    """
    if rom is None:
        return None

    if 70 <= rom <= 130:
        return 100
    elif 50 <= rom < 70 or 130 < rom <= 150:
        return 80
    elif 30 <= rom < 50 or 150 < rom <= 170:
        return 60
    else:
        return 40


def score_symmetry(difference):
    """
    Lower left-right difference receives a higher prototype score.
    """
    if difference is None:
        return None

    if difference <= 5:
        return 100
    elif difference <= 15:
        return 80
    elif difference <= 25:
        return 60
    elif difference <= 40:
        return 50
    else:
        return 40


def assess_sprint(*args, **kwargs):
    """
    Accepts sprint metrics regardless of whether the API passes
    one argument or multiple arguments.

    It searches all supplied arguments instead of silently ignoring
    the second argument.
    """

    sprint_metrics = None

    # Search keyword arguments first.
    for value in kwargs.values():
        sprint_metrics = find_sprint_metrics(value)
        if sprint_metrics is not None:
            break

    # Search every positional argument.
    if sprint_metrics is None:
        for value in args:
            sprint_metrics = find_sprint_metrics(value)
            if sprint_metrics is not None:
                break

    if sprint_metrics is None:
        return {
            "activity": "Sprint",
            "overall_score": None,
            "performance_band": "Not Available",
            "performance_level": "Insufficient Data",
            "assessment_status": "Insufficient Data",
            "score_coverage": 0,
            "available_component_count": 0,
            "unavailable_component_count": 5,
            "available_components": [],
            "unavailable_components": list(WEIGHTS.keys()),
            "component_scores": {
                key: None for key in WEIGHTS
            },
            "metrics": {},
            "recommendations": [
                "Sprint measurements could not be passed to the assessment function."
            ],
            "interpretation": (
                "Sprint measurements were not found in the supplied assessment data."
            ),
            "ml_status": (
                "No validated labelled Sprint dataset or trained Sprint model "
                "is configured. The current assessment uses prototype "
                "rule-based scoring."
            ),
            "prototype_note": (
                "Prototype scores and thresholds are experimental and are "
                "not a professional coaching or medical assessment."
            ),
        }

    knee = sprint_metrics.get("knee_movement", {})
    hip = sprint_metrics.get("hip_movement", {})
    trunk = sprint_metrics.get("trunk_movement", {})

    knee_rom = get_average_rom(knee)
    hip_rom = get_average_rom(hip)
    trunk_rom = get_average_rom(trunk)

    knee_symmetry = get_symmetry_difference(knee)
    hip_symmetry = get_symmetry_difference(hip)

    component_scores = {
        "knee_movement": score_rom(knee_rom),
        "knee_symmetry": score_symmetry(knee_symmetry),
        "hip_movement": score_rom(hip_rom),
        "hip_symmetry": score_symmetry(hip_symmetry),
        "trunk_movement": score_rom(trunk_rom),
    }

    available_components = [
        name for name, score in component_scores.items()
        if score is not None
    ]

    unavailable_components = [
        name for name, score in component_scores.items()
        if score is None
    ]

    available_weight = sum(
        WEIGHTS[name] for name in available_components
    )

    if available_weight > 0:
        overall_score = round(
            sum(
                component_scores[name] * WEIGHTS[name]
                for name in available_components
            ) / available_weight
        )
        score_coverage = round(available_weight / sum(WEIGHTS.values()) * 100)
    else:
        overall_score = None
        score_coverage = 0

    available_count = len(available_components)

    if available_count == 5:
        assessment_status = "Complete"
        performance_level = "Complete"
    elif available_count >= 3:
        assessment_status = "Provisional"
        performance_level = "Provisional"
    else:
        assessment_status = "Insufficient Data"
        performance_level = "Insufficient Data"

    if overall_score is None:
        performance_band = "Not Available"
    elif overall_score >= 85:
        performance_band = "Excellent"
    elif overall_score >= 70:
        performance_band = "Good"
    elif overall_score >= 50:
        performance_band = "Average"
    else:
        performance_band = "Needs Improvement"

    recommendations = []

    if knee_rom is None:
        recommendations.append(
            "Improve video visibility so knee movement can be measured."
        )

    if knee_symmetry is not None and knee_symmetry > 25:
        recommendations.append(
            "Review left-right knee movement differences."
        )

    if hip_rom is None:
        recommendations.append(
            "Ensure the hip landmarks are visible during the sprint."
        )

    if hip_symmetry is not None and hip_symmetry > 25:
        recommendations.append(
            "Review left-right hip movement differences."
        )

    if trunk_rom is None:
        recommendations.append(
            "Ensure the upper body is visible for trunk movement analysis."
        )

    if not recommendations:
        recommendations.append(
            "Continue practising and compare results across future recordings."
        )

    if overall_score is None:
        interpretation = (
            "There is insufficient movement data to generate a Sprint assessment."
        )
    elif assessment_status == "Complete":
        interpretation = (
            "All five prototype measurement components were available. "
            "The result is an experimental rule-based assessment."
        )
    else:
        interpretation = (
            "A provisional Sprint assessment was generated using the "
            "available measurements. Some components could not be assessed."
        )

    return {
        "activity": "Sprint",
        "overall_score": overall_score,
        "performance_band": performance_band,
        "performance_level": performance_level,
        "assessment_status": assessment_status,
        "score_coverage": score_coverage,
        "available_component_count": available_count,
        "unavailable_component_count": len(unavailable_components),
        "available_components": available_components,
        "unavailable_components": unavailable_components,
        "component_scores": component_scores,
        "metrics": {
            "knee_movement": knee,
            "hip_movement": hip,
            "trunk_movement": trunk,
            "knee_angle_variability": sprint_metrics.get(
                "knee_angle_variability"
            ),
        },
        "recommendations": recommendations,
        "interpretation": interpretation,
        "ml_status": (
            "No validated labelled Sprint dataset or trained Sprint model "
            "is configured. The current assessment uses availability-aware "
            "rule-based scoring."
        ),
        "prototype_note": (
            "Scoring thresholds and weights are experimental and have not "
            "been scientifically validated. Knee angle variability is "
            "descriptive only and does not contribute to the overall score. "
            "This is not a medical or professional coaching assessment."
        ),
    }
