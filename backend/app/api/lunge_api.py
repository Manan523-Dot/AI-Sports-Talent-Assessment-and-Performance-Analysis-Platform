
import os
import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException

from backend.app.services.lunge_pose_detection import (
    analyze_lunge_video
)
from backend.app.services.lunge_analysis import (
    analyze_lunge
)
from backend.app.ml.lunge_assessment import (
    assess_lunge
)
from backend.app.database.save_analysis import (
    save_analysis_record
)


router = APIRouter(tags=["Sports Analysis"])

UPLOAD_DIR = "data/uploads"
PROCESSED_DIR = "data/processed"


@router.post("/api/analyze/lunge")
async def analyze_lunge_endpoint(
    video: UploadFile = File(...)
):

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    analysis_id = str(uuid.uuid4())

    input_path = os.path.join(
        UPLOAD_DIR,
        f"{analysis_id}.mp4"
    )

    output_video = os.path.join(
        PROCESSED_DIR,
        f"{analysis_id}_lunge.mp4"
    )

    output_csv = os.path.join(
        PROCESSED_DIR,
        f"{analysis_id}_lunge_features.csv"
    )

    analysis_json = os.path.join(
        PROCESSED_DIR,
        f"{analysis_id}_lunge_analysis.json"
    )

    assessment_json = os.path.join(
        PROCESSED_DIR,
        f"{analysis_id}_lunge_assessment.json"
    )

    processed_video_url = (
        f"/processed/{analysis_id}_lunge.mp4"
    )

    try:

        with open(input_path, "wb") as file:
            file.write(await video.read())

        # Step 1: Pose detection
        analyze_lunge_video(
            input_path,
            output_video,
            output_csv
        )

        # Step 2: Feature analysis
        analyze_lunge(
            output_csv,
            analysis_json
        )

        # Step 3: Assessment
        assessment = assess_lunge(
            analysis_json,
            assessment_json
        )

        # Step 4: Save result in database
        save_analysis_record(
            analysis_id=analysis_id,
            activity="Lunge",
            filename=video.filename,
            processed_video_url=processed_video_url,
            assessment=assessment
        )

        return {
            "success": True,
            "analysis_id": analysis_id,
            "activity": "lunge",
            "video_url": processed_video_url,
            "assessment": assessment
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Lunge analysis failed: {str(error)}"
        )

    finally:

        if os.path.exists(input_path):
            os.remove(input_path)
