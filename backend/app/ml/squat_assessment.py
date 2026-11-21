
import json
from pathlib import Path


# =============================================================
# HELPERS
# =============================================================

def safe_float(value):
    if value is None:
        return None

    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


def safe_int(value, default=0):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def safe_round(value, digits=2):
    if value is None:
        return None

    return round(value, digits)


def get_performance_band(score):
    if score is None:
        return "Insufficient Data"
    if score >= 85:
        return "Strong"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Developing"
    return "Needs Improvement"


# =============================================================
# SQUAT ASSESSMENT
# =============================================================

def assess_squat(input_json, output_json):

    input_json = Path(input_json)
    output_json = Path(output_json)

    with open(input_json, "r", encoding="utf-8") as json_file:
        features = json.load(json_file)

    # =========================================================
    # 1. GET FEATURES
    # =========================================================

    repetitions = safe_int(features.get("repetitions"))

    knee_rom = safe_float(features.get("knee_rom"))
    hip_rom = safe_float(features.get("hip_rom"))

    knee_rom_difference = safe_float(
        features.get("knee_rom_difference")
    )

    hip_rom_difference = safe_float(
        features.get("hip_rom_difference")
    )

    knee_symmetry_available = (
        features.get("knee_symmetry_available", False)
        and knee_rom_difference is not None
    )

    hip_symmetry_available = (
        features.get("hip_symmetry_available", False)
        and hip_rom_difference is not None
    )

    average_bottom_angle = safe_float(
        features.get("average_rep_bottom_angle")
    )

    deepest_rep_angle = safe_float(
        features.get("deepest_rep_angle")
    )

    average_descent_time = safe_float(
        features.get("average_descent_time")
    )

    average_ascent_time = safe_float(
        features.get("average_ascent_time")
    )

    average_rep_time = safe_float(
        features.get("average_rep_time")
    )

    fastest_rep_time = safe_float(
        features.get("fastest_rep_time")
    )

    slowest_rep_time = safe_float(
        features.get("slowest_rep_time")
    )

    rep_time_difference = safe_float(
        features.get("rep_time_difference")
    )

    trunk_angle_change = safe_float(
        features.get("trunk_angle_change")
    )

    rep_details = features.get("rep_details", [])

    if not isinstance(rep_details, list):
        rep_details = []

    # =========================================================
    # 2. COMPONENT SCORES
    # =========================================================

    # No repetitions means there is not enough data to assess.
    if repetitions == 0:

        depth_score = None
        symmetry_score = None
        movement_control_score = None
        tempo_score = None
        consistency_score = None

    else:

        # -----------------------------------------------------
        # DEPTH SCORE
        # -----------------------------------------------------

        if knee_rom is None or average_bottom_angle is None:
            depth_score = None

        elif knee_rom >= 120 and average_bottom_angle <= 95:
            depth_score = 100

        elif knee_rom >= 100 and average_bottom_angle <= 105:
            depth_score = 85

        elif knee_rom >= 80 and average_bottom_angle <= 120:
            depth_score = 70

        else:
            depth_score = 50

        # -----------------------------------------------------
        # SYMMETRY SCORE
        # -----------------------------------------------------

        if not knee_symmetry_available:
            symmetry_score = None

        elif knee_rom_difference <= 5:
            symmetry_score = 100

        elif knee_rom_difference <= 10:
            symmetry_score = 85

        elif knee_rom_difference <= 20:
            symmetry_score = 70

        else:
            symmetry_score = 50

        # -----------------------------------------------------
        # MOVEMENT CONTROL SCORE
        # -----------------------------------------------------

        if trunk_angle_change is None:
            movement_control_score = None

        elif trunk_angle_change <= 20:
            movement_control_score = 100

        elif trunk_angle_change <= 30:
            movement_control_score = 85

        elif trunk_angle_change <= 40:
            movement_control_score = 70

        else:
            movement_control_score = 50

        # -----------------------------------------------------
        # TEMPO SCORE
        # -----------------------------------------------------

        if (
            average_descent_time is None
            or average_ascent_time is None
            or average_descent_time < 0
            or average_ascent_time < 0
            or average_descent_time + average_ascent_time <= 0
        ):
            tempo_score = None

        else:

            descent_ratio = (
                average_descent_time
                / (average_descent_time + average_ascent_time)
            )

            if 0.35 <= descent_ratio <= 0.60:
                tempo_score = 100

            elif 0.25 <= descent_ratio <= 0.70:
                tempo_score = 85

            else:
                tempo_score = 70

        # -----------------------------------------------------
        # CONSISTENCY SCORE
        # -----------------------------------------------------

        if repetitions < 2 or rep_time_difference is None:
            consistency_score = None

        elif rep_time_difference <= 0.20:
            consistency_score = 100

        elif rep_time_difference <= 0.40:
            consistency_score = 85

        elif rep_time_difference <= 0.70:
            consistency_score = 70

        else:
            consistency_score = 50

    # =========================================================
    # 3. OVERALL SCORE AND COVERAGE
    # =========================================================

    component_weights = {
        "depth_score": 0.30,
        "symmetry_score": 0.20,
        "movement_control_score": 0.20,
        "tempo_score": 0.15,
        "consistency_score": 0.15
    }

    component_scores = {
        "depth_score": depth_score,
        "symmetry_score": symmetry_score,
        "movement_control_score": movement_control_score,
        "tempo_score": tempo_score,
        "consistency_score": consistency_score
    }

    available_components = [
        name
        for name, score in component_scores.items()
        if score is not None
    ]

    unavailable_components = [
        name
        for name, score in component_scores.items()
        if score is None
    ]

    available_weight = sum(
        component_weights[name]
        for name in available_components
    )

    if repetitions == 0:
        score_coverage = 0
        performance_score = None

    else:
        score_coverage = round(
            available_weight * 100,
            2
        )

        if available_weight > 0:

            weighted_total = sum(
                component_scores[name] * component_weights[name]
                for name in available_components
            )

            # This is a score based only on measured components.
            # Missing components are excluded, not treated as perfect.
            performance_score = round(
                weighted_total / available_weight,
                2
            )

        else:
            performance_score = None

    # =========================================================
    # 4. ASSESSMENT STATUS
    # =========================================================

    if repetitions == 0:

        assessment_status = "Insufficient Data"

    elif score_coverage < 100:

        assessment_status = "Provisional"

    else:

        assessment_status = "Complete"

    # =========================================================
    # 5. PERFORMANCE BAND
    # =========================================================

    performance_band = get_performance_band(
        performance_score
    )

    if assessment_status == "Insufficient Data":
        movement_level = "Insufficient Data"

    elif assessment_status == "Provisional":
        movement_level = "Provisional"

    else:
        movement_level = performance_band

    # =========================================================
    # 6. SCORE INTERPRETATION
    # =========================================================

    total_components = len(component_weights)
    available_count = len(available_components)

    if assessment_status == "Insufficient Data":

        score_interpretation = (
            "A performance score could not be calculated because "
            "no complete squat repetitions were detected."
        )

    elif assessment_status == "Provisional":

        score_interpretation = (
            f"The provisional performance band is {performance_band}. "
            f"The score uses {available_count} of {total_components} "
            f"assessment components. "
            "Unavailable measurements were excluded from the calculation. "
            "This result should not be treated as a complete assessment."
        )

    else:

        score_interpretation = (
            "All assessment components were available for scoring."
        )

    # =========================================================
    # 7. RECOMMENDATIONS
    # =========================================================

    recommendations = []

    if repetitions == 0:

        recommendations.append(
            "No complete squat repetitions were detected. "
            "Ensure the full movement is visible in the video."
        )

    else:

        # Depth feedback
        if depth_score is None:

            recommendations.append(
                "Squat depth could not be assessed reliably."
            )

        elif depth_score < 85:

            recommendations.append(
                "Work on maintaining a consistent squat range of motion "
                "while keeping the movement controlled."
            )

        else:

            recommendations.append(
                "The detected repetitions show a consistent "
                "squat range of motion."
            )

        # Symmetry feedback
        if symmetry_score is None:

            recommendations.append(
                "Left-right knee symmetry could not be assessed "
                "because both knees were not sufficiently visible "
                "together. This does not mean symmetry is perfect."
            )

        elif symmetry_score < 85:

            recommendations.append(
                "Focus on maintaining similar movement on both "
                "sides during the squat."
            )

        else:

            recommendations.append(
                "Left-right knee movement was reasonably "
                "consistent in the measured frames."
            )

        # Movement control feedback
        if movement_control_score is None:

            recommendations.append(
                "Trunk movement could not be assessed reliably."
            )

        elif movement_control_score < 85:

            recommendations.append(
                "Focus on maintaining controlled trunk movement "
                "throughout the squat."
            )

        else:

            recommendations.append(
                "Trunk movement remained relatively controlled "
                "during the analyzed repetitions."
            )

        # Tempo feedback
        if tempo_score is None:

            recommendations.append(
                "Movement tempo could not be assessed reliably."
            )

        else:

            recommendations.append(
                "Descent and ascent timing was used to assess "
                "movement rhythm."
            )

        # Consistency feedback
        if consistency_score is None:

            recommendations.append(
                "More complete repetitions are needed to assess "
                "repetition-to-repetition consistency."
            )

        elif consistency_score < 85:

            recommendations.append(
                "Try to maintain a more consistent duration "
                "across repetitions."
            )

        else:

            recommendations.append(
                "Repetition timing was assessed for consistency "
                "across the analyzed set."
            )

    if assessment_status == "Provisional":

        recommendations.append(
            "This is a provisional assessment because some measurements "
            "were unavailable. Interpret the performance band alongside "
            "the measurement coverage."
        )

    # =============================================================
    # 8. FINAL ASSESSMENT
    # =============================================================

    assessment = {

        "activity": "squat",

        "performance_score": performance_score,

        "movement_level": movement_level,

        "performance_band": performance_band,

        "assessment_status": assessment_status,

        "score_interpretation": score_interpretation,

        "score_coverage_percent": score_coverage,

        "available_component_count": available_count,

        "total_component_count": total_components,

        "available_components": available_components,

        "unavailable_components": unavailable_components,

        "repetitions": repetitions,

        "metrics": {

            "knee_range_of_motion": safe_round(knee_rom),

            "hip_range_of_motion": safe_round(hip_rom),

            "knee_symmetry_difference": safe_round(
                knee_rom_difference
            ),

            "hip_symmetry_difference": safe_round(
                hip_rom_difference
            ),

            "knee_symmetry_available": knee_symmetry_available,

            "knee_symmetry_frames": features.get(
                "knee_symmetry_frames", 0
            ),

            "hip_symmetry_available": hip_symmetry_available,

            "hip_symmetry_frames": features.get(
                "hip_symmetry_frames", 0
            ),

            "detected_sides": features.get(
                "detected_sides", "unknown"
            ),

            "visibility_counts": features.get(
                "visibility_counts", {}
            ),

            "average_bottom_angle": safe_round(
                average_bottom_angle
            ),

            "deepest_rep_angle": safe_round(
                deepest_rep_angle
            ),

            "descent_time": safe_round(
                average_descent_time
            ),

            "ascent_time": safe_round(
                average_ascent_time
            ),

            "average_rep_time": safe_round(
                average_rep_time
            ),

            "fastest_rep_time": safe_round(
                fastest_rep_time
            ),

            "slowest_rep_time": safe_round(
                slowest_rep_time
            ),

            "rep_time_difference": safe_round(
                rep_time_difference
            ),

            "trunk_angle_change": safe_round(
                trunk_angle_change
            )
        },

        "component_scores": component_scores,

        "rep_details": rep_details,

        "recommendations": recommendations
    }

    # =============================================================
    # 9. SAVE ASSESSMENT
    # =============================================================

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_json, "w", encoding="utf-8") as json_file:

        json.dump(
            assessment,
            json_file,
            indent=4
        )

    # =============================================================
    # 10. CONSOLE SUMMARY
    # =============================================================

    print("\nSQUAT ASSESSMENT COMPLETED")
    print("=" * 55)

    print("Repetitions:", repetitions)
    print("Performance score:", performance_score)
    print("Performance band:", performance_band)
    print("Movement level:", movement_level)
    print("Assessment status:", assessment_status)
    print("Score coverage:", score_coverage, "%")

    print("\nComponent scores:")

    for name, score in component_scores.items():
        print(f"{name}: {score}")

    print("\nRecommendations:")

    for recommendation in recommendations:
        print("-", recommendation)

    print("\nAssessment saved:", output_json)

    return assessment


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parents[3]

    input_file = (
        project_dir
        / "data"
        / "processed"
        / "squat_features.json"
    )

    output_file = (
        project_dir
        / "data"
        / "processed"
        / "squat_assessment.json"
    )

    assess_squat(
        input_file,
        output_file
    )
