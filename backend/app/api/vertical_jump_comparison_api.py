import os
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.services.pose_detection import analyze_video
from backend.app.services.vertical_jump_analysis import analyze_vertical_jump
from backend.app.ml.vertical_jump_assessment import assess_vertical_jump
from backend.app.ml.vertical_jump_comparison_engine import (
    build_vertical_jump_comparison,
)
from backend.app.database.save_analysis import save_analysis_record


router = APIRouter(
    prefix="/api/compare",
    tags=["Video Comparison"]
)

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}


def _process_vertical_jump_video(upload, label):
    analysis_id = str(uuid.uuid4())

    extension = os.path.splitext(
        upload.filename or ""
    )[1].lower()

    upload_dir = "data/uploads"
    processed_dir = "data/processed"

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    input_path = os.path.join(
        upload_dir,
        f"{analysis_id}_vertical_jump_comparison{extension}"
    )

    output_video = os.path.join(
        processed_dir,
        f"{analysis_id}_vertical_jump_pose.mp4"
    )

    output_csv = os.path.join(
        processed_dir,
        f"{analysis_id}_vertical_jump_angles.csv"
    )

    output_features = os.path.join(
        processed_dir,
        f"{analysis_id}_vertical_jump_features.json"
    )

    output_assessment = os.path.join(
        processed_dir,
        f"{analysis_id}_vertical_jump_assessment.json"
    )

    # 1. Save uploaded video.
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    # 2. Reuse the existing pose-detection pipeline.
    analyze_video(
        input_path,
        output_video,
        output_csv
    )

    # 3. Reuse the existing Vertical Jump analysis.
    features = analyze_vertical_jump(
        output_csv,
        output_features
    )

    processed_video_url = (
        f"/processed/{analysis_id}_vertical_jump_pose.mp4"
    )

    # 4. Build the same base assessment structure as the
    #    existing single-video Vertical Jump API.
    assessment = {
        "activity": "vertical_jump",
        "assessment_method": "pose_analysis",
        "jump_detected": features.get(
            "jump_detected",
            False
        ),
        "metrics": {
            "estimated_jump_height_cm": features.get(
                "estimated_jump_height_cm"
            ),
            "flight_time": features.get("flight_time"),
            "candidate_flight_time_seconds": features.get(
                "candidate_flight_time_seconds"
            ),
            "takeoff_time": features.get("takeoff_time"),
            "landing_time": features.get("landing_time"),
            "knee_range_of_motion": features.get(
                "knee_range_of_motion"
            ),
            "knee_rom_difference": features.get(
                "knee_rom_difference"
            ),
            "takeoff_knee_angle": features.get(
                "takeoff_knee_angle"
            ),
            "landing_knee_angle": features.get(
                "landing_knee_angle"
            ),
            "bottom_knee_angle": features.get(
                "bottom_knee_angle"
            ),
        },
        "recommendations": [
            "This is a preliminary pose-based analysis.",
            "Jump height is a physics-based estimate, not a direct measurement.",
        ],
        "assessment_status": "Provisional",
        "score_coverage": 0,
    }

    # 5. Run the existing assessment only when a candidate
    #    jump cycle has been detected.
    if features.get("jump_detected", False):
        try:
            ml_assessment = assess_vertical_jump(
                output_features,
                output_assessment
            )

            assessment.update({
                "assessment_method": ml_assessment.get(
                    "assessment_method",
                    "machine_learning"
                ),
                "model_name": ml_assessment.get("model_name"),
                "model_version": ml_assessment.get("model_version"),
                "performance_score": ml_assessment.get(
                    "performance_score"
                ),
                "performance_level": ml_assessment.get(
                    "performance_level"
                ),
                "component_scores": ml_assessment.get(
                    "component_scores",
                    {}
                ),
                "model_evaluation": ml_assessment.get(
                    "model_evaluation",
                    {}
                ),
                "recommendations": ml_assessment.get(
                    "recommendations",
                    []
                ),
                "assessment_status": "Complete",
                "score_coverage": 100,
            })

        except (
            FileNotFoundError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            assessment["assessment_status"] = "Provisional"
            assessment["score_coverage"] = 0
            assessment["assessment_note"] = (
                "Movement measurements were generated, but "
                f"the assessment could not be calculated: {str(exc)}"
            )

    else:
        assessment["assessment_note"] = features.get(
            "detection_note",
            "A valid candidate jump cycle was not detected."
        )

    return {
        "filename": upload.filename,
        "processed_video_url": processed_video_url,
        "assessment": assessment,
        "features": features,
        "label": label,
    }


@router.post("/vertical-jump")
async def compare_vertical_jump_videos(
    video_a: UploadFile = File(...),
    video_b: UploadFile = File(...),
    comparison_mode: str = Form("progress"),
):
    if comparison_mode not in {"progress", "reference"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "comparison_mode must be "
                "'progress' or 'reference'."
            ),
        )

    for upload in (video_a, video_b):
        if not upload.filename:
            raise HTTPException(
                status_code=400,
                detail="Please upload both Vertical Jump videos."
            )

        extension = os.path.splitext(
            upload.filename
        )[1].lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported video format. "
                    "Upload MP4, AVI, MOV or MKV."
                ),
            )

    comparison_id = str(uuid.uuid4())

    try:
        # Process both videos independently.
        first = _process_vertical_jump_video(
            video_a,
            "A"
        )

        second = _process_vertical_jump_video(
            video_b,
            "B"
        )

        # Build the comparison using extracted features
        # and the two existing assessment results.
        comparison = build_vertical_jump_comparison(
            first["assessment"],
            second["assessment"],
            first["features"],
            second["features"],
            comparison_mode,
        )

        result = {
            "success": True,
            "comparison_id": comparison_id,
            "exercise": "Vertical Jump",
            "comparison_mode": comparison_mode,

            "video_a": {
                "filename": first["filename"],
                "processed_video_url": first[
                    "processed_video_url"
                ],
                "assessment": first["assessment"],
                "features": first["features"],
                "knee_angle_curve": (
                    _get_curve_for_video(
                        first["features"]
                    )
                ),
            },

            "video_b": {
                "filename": second["filename"],
                "processed_video_url": second[
                    "processed_video_url"
                ],
                "assessment": second["assessment"],
                "features": second["features"],
                "knee_angle_curve": (
                    _get_curve_for_video(
                        second["features"]
                    )
                ),
            },

            "comparison": comparison,
            "insights": comparison["observations"],

            "report": {
                "metrics": comparison["metrics"],
                "observations": comparison["observations"],
                "limitations": comparison["limitations"],
            },

            "limitations": comparison["limitations"],
        }

        # Save the complete comparison in existing Analysis History.
        save_analysis_record(
            analysis_id=comparison_id,
            activity="Vertical Jump Comparison",
            filename=(
                f"{first['filename']} vs "
                f"{second['filename']}"
            ),
            processed_video_url=first["processed_video_url"],
            assessment=result,
        )

        return result

    except HTTPException:
        raise

    except Exception as error:
        print(
            "VERTICAL JUMP COMPARISON ERROR:",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Vertical Jump comparison failed: "
                f"{str(error)}"
            ),
        )

    finally:
        await video_a.close()
        await video_b.close()


def _get_curve_for_video(features):
    """Return the normalized knee-angle curve for one video."""
    from backend.app.ml.vertical_jump_comparison_engine import (
        _normalized_curve,
    )

    return _normalized_curve(features)