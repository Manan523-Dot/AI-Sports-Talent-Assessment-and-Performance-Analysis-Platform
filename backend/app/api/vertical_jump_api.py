import os
import shutil
import uuid

from fastapi import APIRouter, File, UploadFile

from backend.app.services.pose_detection import analyze_video
from backend.app.services.vertical_jump_analysis import analyze_vertical_jump
from backend.app.ml.vertical_jump_assessment import assess_vertical_jump
from backend.app.database.save_analysis import save_analysis_record


router = APIRouter(
    prefix="/api/analyze",
    tags=["Sports Analysis"]
)


@router.post("/vertical-jump")
async def analyze_vertical_jump_video(
    video: UploadFile = File(...)
):
    analysis_id = str(uuid.uuid4())

    input_video = f"data/uploads/{analysis_id}.mp4"
    output_video = f"data/processed/{analysis_id}_vertical_jump_pose.mp4"
    output_csv = f"data/processed/{analysis_id}_vertical_jump_angles.csv"
    output_features = f"data/processed/{analysis_id}_vertical_jump_features.json"
    output_assessment = f"data/processed/{analysis_id}_vertical_jump_assessment.json"

    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    try:
        # 1. Save uploaded video
        with open(input_video, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 2. Detect pose and extract knee angles
        analyze_video(input_video, output_video, output_csv)

        # 3. Analyze movement and calculate measurement features
        features = analyze_vertical_jump(output_csv, output_features)

        processed_video_url = f"/processed/{analysis_id}_vertical_jump_pose.mp4"

        # Base result: a measurement-only / provisional assessment.
        assessment = {
            "activity": "vertical_jump",
            "assessment_method": "pose_analysis",
            "jump_detected": features.get("jump_detected", False),
            "metrics": {
                "estimated_jump_height_cm": features.get("estimated_jump_height_cm"),
                "flight_time": features.get("flight_time"),
                "candidate_flight_time_seconds": features.get("candidate_flight_time_seconds"),
                "takeoff_time": features.get("takeoff_time"),
                "landing_time": features.get("landing_time"),
                "knee_range_of_motion": features.get("knee_range_of_motion"),
                "knee_rom_difference": features.get("knee_rom_difference"),
                "takeoff_knee_angle": features.get("takeoff_knee_angle"),
                "landing_knee_angle": features.get("landing_knee_angle"),
            },
            "recommendations": [
                "This is a preliminary pose-based analysis.",
                "Jump height is a physics-based estimate, not a direct measurement."
            ],
            "assessment_status": "Provisional",
            "score_coverage": 0,
        }

        # 4. Run the trained ML assessment only if the movement analysis
        #    marked the flight interval as valid.
        if features.get("jump_detected", False):
            try:
                ml_assessment = assess_vertical_jump(
                    output_features,
                    output_assessment
                )

                # Keep the API's existing metrics, but add the ML score,
                # category, model details, and personalized recommendations.
                assessment.update({
                    "assessment_method": ml_assessment.get(
                        "assessment_method", "machine_learning"
                    ),
                    "model_name": ml_assessment.get("model_name"),
                    "model_version": ml_assessment.get("model_version"),
                    "performance_score": ml_assessment.get("performance_score"),
                    "performance_level": ml_assessment.get("performance_level"),
                    "component_scores": ml_assessment.get("component_scores", {}),
                    "model_evaluation": ml_assessment.get("model_evaluation", {}),
                    "recommendations": ml_assessment.get("recommendations", []),
                    "assessment_status": "Complete",
                    "score_coverage": 100,
                })

            except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
                # Do not invent a score if the model isn't available or
                # assessment can't run. Return measurements with clear status.
                assessment["assessment_status"] = "Provisional"
                assessment["score_coverage"] = 0
                assessment["assessment_note"] = (
                    "The movement measurements were generated, but the ML score "
                    f"could not be calculated: {str(exc)}"
                )
        else:
            assessment["assessment_note"] = features.get(
                "detection_note",
                "A valid flight interval was not confirmed, so no ML score was generated."
            )

        # 5. Save assessment to history
        save_analysis_record(
            analysis_id=analysis_id,
            activity="Vertical Jump",
            filename=video.filename,
            processed_video_url=processed_video_url,
            assessment=assessment
        )

        # 6. Return result to frontend
        return {
            "analysis_id": analysis_id,
            "filename": video.filename,
            "processed_video_url": processed_video_url,
            "assessment": assessment,
            "knee_angle_data": features.get("knee_angle_data", []),
            "detection_status": features.get("detection_status"),
            "detection_note": features.get("detection_note"),
        }

    finally:
        await video.close()
