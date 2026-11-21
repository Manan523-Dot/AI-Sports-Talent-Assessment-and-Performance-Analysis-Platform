import json
import os


def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, value))


def assess_lunge(input_json, output_json):

    with open(input_json, "r", encoding="utf-8") as file:
        data = json.load(file)

    reps = data.get("detected_repetitions", 0)
    coverage = data.get("coverage", 0)
    rom = data.get("average_knee_rom")
    consistency = data.get("rep_consistency")
    torso_angle = data.get("average_torso_angle")

    # --------------------------------------------------
    # 1. DEPTH SCORE
    # --------------------------------------------------

    depth_score = (
        clamp((rom / 75) * 100)
        if rom is not None
        else None
    )

    # --------------------------------------------------
    # 2. REP CONSISTENCY SCORE
    # --------------------------------------------------

    consistency_score = (
        clamp(consistency)
        if consistency is not None
        else None
    )

    # --------------------------------------------------
    # 3. VISIBILITY SCORE
    # --------------------------------------------------

    visibility_score = clamp(coverage)

    # --------------------------------------------------
    # 4. TORSO CONTROL SCORE
    # --------------------------------------------------

    torso_score = None

    if torso_angle is not None:

        torso_deviation = abs(180 - torso_angle)

        # Small deviations are tolerated.
        # Larger deviations gradually reduce the score.
        torso_score = clamp(
            100 - max(0, torso_deviation - 8) * 2
        )

    # --------------------------------------------------
    # COMPONENT SCORES
    # --------------------------------------------------

    components = {}

    if depth_score is not None:
        components["depth"] = round(
            depth_score, 2
        )

    if consistency_score is not None:
        components["consistency"] = round(
            consistency_score, 2
        )

    components["visibility"] = round(
        visibility_score, 2
    )

    if torso_score is not None:
        components["torso_control"] = round(
            torso_score, 2
        )

    # --------------------------------------------------
    # WEIGHTED PERFORMANCE SCORE
    # --------------------------------------------------

    weights = {
        "depth": 0.40,
        "consistency": 0.25,
        "visibility": 0.15,
        "torso_control": 0.20
    }

    available = {
        key: value
        for key, value in components.items()
        if key in weights and value is not None
    }

    total_weight = sum(
        weights[key]
        for key in available
    )

    if total_weight > 0:

        score = sum(
            available[key] * weights[key]
            for key in available
        ) / total_weight

    else:
        score = 0

    score = round(score, 2)

    # --------------------------------------------------
    # PERFORMANCE STATUS
    # --------------------------------------------------

    if score >= 85:
        status = "Excellent"

    elif score >= 70:
        status = "Good"

    elif score >= 50:
        status = "Average"

    else:
        status = "Needs Improvement"

    # --------------------------------------------------
    # ASSESSMENT RELIABILITY
    # --------------------------------------------------

    assessment_status = (
        "Complete"
        if coverage >= 70 and reps >= 3
        else "Limited"
    )

    # --------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------

    recommendations = []

    if depth_score is not None and depth_score < 70:

        recommendations.append(
            "Focus on controlled lunge depth while "
            "maintaining comfortable movement."
        )

    if consistency_score is not None and consistency_score < 70:

        recommendations.append(
            "Maintain a consistent pace across repetitions."
        )

    if torso_score is not None and torso_score < 65:

        recommendations.append(
            "Try maintaining a more upright and controlled "
            "torso position during the movement."
        )

    if coverage < 70:

        recommendations.append(
            "Improve camera framing and body visibility "
            "for more reliable analysis."
        )

    if reps < 3:

        recommendations.append(
            "Perform at least three complete repetitions "
            "for a more reliable assessment."
        )

    if not recommendations:

        recommendations.append(
            "Your measured movement metrics are consistent. "
            "Continue practicing with controlled repetitions."
        )

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    result = {
        **data,
        "performance_score": score,
        "performance_status": status,
        "assessment_status": assessment_status,
        "assessment_components": components,
        "recommendations": recommendations
    }

    output_directory = os.path.dirname(output_json)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print("\nLUNGE ASSESSMENT COMPLETED")
    print("Score:", score)
    print("Status:", status)
    print("Assessment:", assessment_status)
    print("Components:", components)

    return result