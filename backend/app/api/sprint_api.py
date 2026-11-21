
import os
import shutil
import uuid
import json

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.services.pose_detection import analyze_video
from backend.app.services.sprint_analysis import analyze_sprint
from backend.app.ml.sprint_assessment import assess_sprint
from backend.app.database.save_analysis import save_analysis_record


router = APIRouter(
    prefix="/api/analyze",
    tags=["Sports Analysis"],
)


@router.post("/sprint")
async def analyze_sprint_video(video: UploadFile = File(...)):

    if not video.filename:
        raise HTTPException(
            status_code=400,
            detail="Please upload a Sprint video.",
        )

    extension = os.path.splitext(video.filename)[1].lower()

    if extension not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Upload MP4, AVI, MOV or MKV.",
        )

    analysis_id = str(uuid.uuid4())

    upload_dir = "data/uploads"
    processed_dir = "data/processed"

    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    input_video = os.path.join(
        upload_dir,
        f"{analysis_id}{extension}",
    )

    output_video = os.path.join(
        processed_dir,
        f"{analysis_id}_sprint_pose.mp4",
    )

    output_csv = os.path.join(
        processed_dir,
        f"{analysis_id}_sprint_angles.csv",
    )

    output_features = os.path.join(
        processed_dir,
        f"{analysis_id}_sprint_features.json",
    )

    output_assessment = os.path.join(
        processed_dir,
        f"{analysis_id}_sprint_assessment.json",
    )

    try:
        with open(input_video, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Step 1: Pose detection and angle extraction.
        analyze_video(
            input_video,
            output_video,
            output_csv,
        )

        # Step 2: Extract Sprint movement features.
        features = analyze_sprint(
            output_csv,
            output_features,
        )

        # Step 3: Pass actual measurement dictionary to assessment.
        assessment = assess_sprint(features)

        # Step 4: Save assessment JSON.
        with open(output_assessment, "w", encoding="utf-8") as file:
            json.dump(assessment, file, indent=4)

        processed_video_url = (
            f"/processed/{analysis_id}_sprint_pose.mp4"
        )

        # Step 5: Preserve database history integration.
        save_analysis_record(
            analysis_id=analysis_id,
            activity="Sprint",
            filename=video.filename,
            processed_video_url=processed_video_url,
            assessment=assessment,
        )

        return {
            "analysis_id": analysis_id,
            "filename": video.filename,
            "processed_video_url": processed_video_url,

            "assessment": assessment,

            "knee_angle_data": features.get(
                "knee_angle_data",
                [],
            ),

            "measurement_visibility": {
                "detected_sides": features.get(
                    "detected_sides",
                    {},
                ),
                "visibility_counts": features.get(
                    "visibility_counts",
                    {},
                ),
                "feature_availability": features.get(
                    "feature_availability",
                    {},
                ),
            },

            "sprint_metrics": {
                "knee_movement": features.get("knee_movement"),
                "hip_movement": features.get("hip_movement"),
                "trunk_movement": features.get("trunk_movement"),
                "duration_seconds": features.get("duration_seconds"),
                "frame_count": features.get("frame_count"),
            },
        }

    except HTTPException:
        raise

    except Exception as error:
        print("SPRINT ANALYSIS ERROR:", str(error))

        raise HTTPException(
            status_code=500,
            detail=f"Sprint analysis failed: {str(error)}",
        )

    finally:
        await video.close()
