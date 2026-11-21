
import json
from pathlib import Path


COMPONENT_WEIGHTS = {
    "depth_score": 0.30,
    "alignment_score": 0.25,
    "symmetry_score": 0.20,
    "tempo_score": 0.15,
    "consistency_score": 0.10
}


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_round(value, digits=2):
    value = safe_float(value)
    return round(value, digits) if value is not None else None


def get_performance_band(score):
    if score is None:
        return "Insufficient Data"
    elif score >= 85:
        return "Strong"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Developing"
    return "Needs Improvement"


def assess_pushup(input_json, output_json):

    input_json = Path(input_json)
    output_json = Path(output_json)

    if not input_json.exists():
        raise FileNotFoundError(
            f"Push-up feature file not found: {input_json}"
        )

    with open(input_json, "r", encoding="utf-8") as file:
        features = json.load(file)

    repetitions = int(features.get("repetitions", 0))

    metrics = features.get("metrics", {})
    original_scores = features.get("component_scores", {})

    # Validate and collect available component scores.
    component_scores = {}

    for name in COMPONENT_WEIGHTS:

        score = safe_float(original_scores.get(name))

        if score is not None and 0 <= score <= 100:
            component_scores[name] = score
        else:
            component_scores[name] = None

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

    # Calculate normalized performance score.
    performance_score = (
        round(
            sum(
                component_scores[name] * COMPONENT_WEIGHTS[name]
                for name in available
            ) / available_weight,
            2
        )
        if repetitions > 0 and available_weight > 0
        else None
    )

    # Assessment status.
    if repetitions == 0 or performance_score is None:
        status = "Insufficient Data"
    elif coverage < 100:
        status = "Provisional"
    else:
        status = "Complete"

    performance_band = get_performance_band(performance_score)

    movement_level = (
        "Insufficient Data"
        if status == "Insufficient Data"
        else "Provisional"
        if status == "Provisional"
        else performance_band
    )

    # Generate assessment recommendations.
    recommendations = []

    if repetitions == 0:

        recommendations.append(
            "No complete push-up repetitions were detected. "
            "Ensure the full body and movement are visible."
        )

    else:

        recommendation_map = {
            "depth_score": (
                "Work on maintaining consistent push-up depth."
            ),
            "alignment_score": (
                "Focus on maintaining a straighter body line "
                "throughout each repetition."
            ),
            "symmetry_score": (
                "Focus on maintaining more consistent left-right "
                "elbow movement."
            ),
            "tempo_score": (
                "Work on maintaining a controlled and consistent "
                "movement tempo."
            ),
            "consistency_score": (
                "Try to maintain a more consistent duration "
                "across repetitions."
            )
        }

        positive_map = {
            "depth_score": (
                "Good elbow depth was observed in the detected "
                "repetitions."
            ),
            "alignment_score": (
                "Body alignment remained relatively consistent."
            ),
            "symmetry_score": (
                "Left-right elbow movement was reasonably consistent."
            ),
            "tempo_score": (
                "Repetition tempo was within the selected target range."
            ),
            "consistency_score": (
                "Repetition timing was reasonably consistent."
            )
        }

        for name in COMPONENT_WEIGHTS:

            score = component_scores[name]

            if score is None:
                recommendations.append(
                    f"{name.replace('_', ' ').capitalize()} "
                    "could not be assessed reliably."
                )

            elif score < 85:
                recommendations.append(
                    recommendation_map[name]
                )

            else:
                recommendations.append(
                    positive_map[name]
                )

    if status == "Provisional":
        recommendations.append(
            "This assessment is provisional because some "
            "measurement components were unavailable."
        )

    # Score interpretation.
    score_interpretation = (
        "Insufficient movement data for a reliable assessment."
        if performance_score is None
        else "Strong performance across the measured components."
        if performance_score >= 85
        else "Good performance with some areas requiring improvement."
        if performance_score >= 70
        else "Developing performance. Several components need attention."
        if performance_score >= 55
        else "More movement control and consistency are needed."
    )

    assessment = {
        "activity": "pushup",
        "performance_score": performance_score,
        "movement_level": movement_level,
        "performance_band": performance_band,
        "assessment_status": status,

        "score_coverage_percent": coverage,
        "available_component_count": len(available),
        "total_component_count": len(COMPONENT_WEIGHTS),

        "available_components": available,
        "unavailable_components": unavailable,

        "repetitions": repetitions,

        "score_interpretation": score_interpretation,

        "metrics": {
            "elbow_range_of_motion": safe_round(
                metrics.get("elbow_range_of_motion")
            ),
            "average_bottom_elbow_angle": safe_round(
                metrics.get("average_bottom_elbow_angle")
            ),
            "body_alignment_deviation": safe_round(
                metrics.get("body_alignment_deviation")
            ),
            "elbow_symmetry_difference": safe_round(
                metrics.get("elbow_symmetry_difference")
            ),
            "elbow_symmetry_available": metrics.get(
                "elbow_symmetry_available", False
            ),
            "average_rep_time": safe_round(
                metrics.get("average_rep_time")
            ),
            "rep_time_difference": safe_round(
                metrics.get("rep_time_difference")
            )
        },

        "component_scores": component_scores,

        "rep_details": features.get("rep_details", []),

        "recommendations": recommendations
    }

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            assessment,
            file,
            indent=4
        )

    print("\nPUSH-UP ASSESSMENT COMPLETED")
    print("Repetitions:", repetitions)
    print("Performance score:", performance_score)
    print("Performance band:", performance_band)
    print("Assessment status:", status)
    print("Score coverage:", coverage, "%")

    return assessment


if __name__ == "__main__":

    assess_pushup(
        "data/processed/pushup_features.json",
        "data/processed/pushup_assessment.json"
    )
