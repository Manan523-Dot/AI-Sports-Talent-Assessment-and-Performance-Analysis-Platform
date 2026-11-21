
import os
import json
import joblib
import pandas as pd


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "vertical_jump",
    "random_forest_model.joblib"
)


def get_performance_level(score):
    if score >= 85:
        return "Strong"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Developing"
    return "Needs Improvement"


def create_fallback_assessment(features):
    """
    Creates a provisional movement-analysis score when the
    trained Random Forest model is unavailable.

    IMPORTANT:
    This score measures measurement completeness and
    left-right movement consistency. It is NOT a validated
    athletic performance score.
    """

    jump_detected = bool(features.get("jump_detected", False))

    jump_height = float(
        features.get("estimated_jump_height_cm", 0) or 0
    )

    flight_time = float(
        features.get("flight_time", 0) or 0
    )

    knee_rom = float(
        features.get("knee_range_of_motion", 0) or 0
    )

    symmetry_difference = float(
        features.get("knee_rom_difference", 0) or 0
    )

    left_rom = float(features.get("left_knee_rom", 0) or 0)
    right_rom = float(features.get("right_knee_rom", 0) or 0)

    takeoff_angle = float(
        features.get("takeoff_knee_angle", 0) or 0
    )

    landing_angle = float(
        features.get("landing_knee_angle", 0) or 0
    )

    recommendations = []

    # Provisional measurement-completeness score.
    # This does not represent athletic ability.
    score = 0.0

    if jump_detected:
        score += 40
    else:
        recommendations.append(
            "A complete jump cycle was not detected. "
            "Use a clear recording with the full movement visible."
        )

    if flight_time > 0 and jump_height > 0:
        score += 20
    else:
        recommendations.append(
            "Jump measurements are incomplete. "
            "Record the full takeoff and landing movement."
        )

    if knee_rom > 0:
        score += 20
    else:
        recommendations.append(
            "Knee range of motion could not be measured reliably."
        )

    # Only assess symmetry when both measurements are available.
    if left_rom > 0 and right_rom > 0:
        if symmetry_difference <= 5:
            score += 20
            recommendations.append(
                "Left and right knee movement appeared relatively consistent."
            )
        elif symmetry_difference <= 10:
            score += 15
            recommendations.append(
                "A small difference was observed between left and right knee movement."
            )
        elif symmetry_difference <= 20:
            score += 10
            recommendations.append(
                "A noticeable left-right knee movement difference was observed."
            )
        else:
            recommendations.append(
                "A larger left-right knee movement difference was observed."
            )
    else:
        recommendations.append(
            "Bilateral knee symmetry could not be assessed because "
            "both knee measurements were not available."
        )

    if jump_detected and flight_time > 0:
        recommendations.append(
            "A candidate flight phase was detected. "
            "Actual takeoff and landing were not independently confirmed."
        )

    if not recommendations:
        recommendations.append(
            "Movement measurements were successfully processed."
        )

    score = round(max(0, min(100, score)), 1)

    return {
        "activity": "vertical_jump",
        "assessment_method": "provisional_rule_based",
        "model_name": None,
        "model_version": None,

        # Kept for compatibility with the existing frontend.
        # This is NOT an ML athletic-performance prediction.
        "performance_score": score,
        "performance_level": get_performance_level(score),

        "score_type": "provisional_measurement_and_consistency_score",
        "score_explanation": (
            "This provisional score reflects measurement completeness "
            "and available left-right knee consistency. It is not a "
            "validated athletic performance rating or an ML prediction."
        ),

        "jump_detected": jump_detected,

        "metrics": {
            "estimated_jump_height_cm": round(jump_height, 2),
            "flight_time": round(flight_time, 3),
            "knee_range_of_motion": round(knee_rom, 2),
            "knee_symmetry_difference": round(symmetry_difference, 2),
            "takeoff_knee_angle": round(takeoff_angle, 2),
            "landing_knee_angle": round(landing_angle, 2)
        },

        "component_scores": {
            "jump_detection": 40 if jump_detected else 0,
            "measurement_availability": (
                (20 if flight_time > 0 and jump_height > 0 else 0)
                + (20 if knee_rom > 0 else 0)
            ),
            "knee_consistency": (
                score - (
                    40 if jump_detected else 0
                ) - (
                    20 if flight_time > 0 and jump_height > 0 else 0
                ) - (
                    20 if knee_rom > 0 else 0
                )
            )
        },

        "model_evaluation": None,
        "recommendations": recommendations
    }


def assess_vertical_jump(input_json, output_json):

    print("\n===== VERTICAL JUMP ASSESSMENT =====")

    if not os.path.exists(input_json):
        raise FileNotFoundError(
            f"Input feature file not found: {input_json}"
        )

    with open(input_json, "r", encoding="utf-8") as file:
        features = json.load(file)

    # --------------------------------------------------
    # FALLBACK: MODEL NOT AVAILABLE
    # --------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print("Random Forest model not found.")
        print("Using provisional rule-based assessment.")

        assessment = create_fallback_assessment(features)

    # --------------------------------------------------
    # EXISTING RANDOM FOREST MODEL
    # --------------------------------------------------

    else:

        model_data = joblib.load(MODEL_PATH)

        model = model_data["model"]
        feature_columns = model_data["feature_columns"]

        model_input = {
            "estimated_jump_height_cm":
                features.get("estimated_jump_height_cm", 0),

            "flight_time":
                features.get("flight_time", 0),

            "knee_range_of_motion":
                features.get("knee_range_of_motion", 0),

            "knee_rom_difference":
                features.get("knee_rom_difference", 0),

            "takeoff_knee_angle":
                features.get("takeoff_knee_angle", 0),

            "landing_knee_angle":
                features.get("landing_knee_angle", 0),

            "max_knee_angle":
                features.get("max_knee_angle", 0),

            "min_knee_angle":
                features.get("min_knee_angle", 0),

            "left_knee_rom":
                features.get("left_knee_rom", 0),

            "right_knee_rom":
                features.get("right_knee_rom", 0),

            "jump_detected":
                int(features.get("jump_detected", False))
        }

        input_df = pd.DataFrame(
            [model_input],
            columns=feature_columns
        )

        predicted_score = float(model.predict(input_df)[0])

        overall_score = round(
            max(0, min(100, predicted_score)), 1
        )

        jump_detected = bool(
            features.get("jump_detected", False)
        )

        jump_height = float(
            features.get("estimated_jump_height_cm", 0) or 0
        )

        flight_time = float(
            features.get("flight_time", 0) or 0
        )

        knee_rom = float(
            features.get("knee_range_of_motion", 0) or 0
        )

        symmetry_difference = float(
            features.get("knee_rom_difference", 0) or 0
        )

        takeoff_angle = float(
            features.get("takeoff_knee_angle", 0) or 0
        )

        landing_angle = float(
            features.get("landing_knee_angle", 0) or 0
        )

        recommendations = []

        if not jump_detected:
            recommendations.append(
                "A complete vertical jump was not clearly detected. "
                "Check the recording and pose detection."
            )
        else:
            if symmetry_difference > 10:
                recommendations.append(
                    "A noticeable difference was observed "
                    "between left and right knee movement."
                )
            else:
                recommendations.append(
                    "Left-right knee movement appeared relatively consistent."
                )

            if flight_time > 0:
                recommendations.append(
                    "A candidate flight phase was detected and "
                    "used in the assessment."
                )

            if knee_rom > 0:
                recommendations.append(
                    "Knee range of motion was included in the ML prediction."
                )

        assessment = {
            "activity": "vertical_jump",
            "assessment_method": "machine_learning",
            "model_name": "Random Forest Regressor",
            "model_version": model_data.get("model_version", "1.0"),

            "performance_score": overall_score,
            "performance_level": get_performance_level(overall_score),

            "score_type": "random_forest_prediction",
            "score_explanation": (
                "Predicted by the trained Random Forest model. "
                "Interpretation depends on the quality and representativeness "
                "of its labelled training dataset."
            ),

            "jump_detected": jump_detected,

            "metrics": {
                "estimated_jump_height_cm": round(jump_height, 2),
                "flight_time": round(flight_time, 3),
                "knee_range_of_motion": round(knee_rom, 2),
                "knee_symmetry_difference": round(symmetry_difference, 2),
                "takeoff_knee_angle": round(takeoff_angle, 2),
                "landing_knee_angle": round(landing_angle, 2)
            },

            "component_scores": {},

            "model_evaluation": {
                "training_samples": model_data.get("training_samples"),
                "mae": model_data.get("mae"),
                "r2_score": model_data.get("r2_score")
            },

            "recommendations": recommendations
        }

    # --------------------------------------------------
    # SAVE ASSESSMENT
    # --------------------------------------------------

    output_directory = os.path.dirname(output_json)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(assessment, file, indent=4)

    print(
        f"Assessment Score: {assessment['performance_score']}"
    )
    print(
        f"Performance Level: {assessment['performance_level']}"
    )
    print(
        f"Assessment Method: {assessment['assessment_method']}"
    )
    print(f"Assessment saved to: {output_json}")

    print("====================================\n")

    return assessment
