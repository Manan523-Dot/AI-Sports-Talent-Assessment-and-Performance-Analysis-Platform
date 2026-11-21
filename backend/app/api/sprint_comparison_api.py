
import json
import os
import shutil
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.services.pose_detection import analyze_video
from backend.app.services.sprint_analysis import analyze_sprint
from backend.app.ml.sprint_assessment import assess_sprint
from backend.app.ml.sprint_comparison_engine import build_sprint_comparison
from backend.app.database.save_analysis import save_analysis_record


router = APIRouter(
    prefix="/api/compare",
    tags=["Video Comparison"]
)

UPLOAD_DIR = "data/uploads"
PROCESSED_DIR = "data/processed"

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


def _read_json(path):
    if not os.path.exists(path):
        raise RuntimeError(
            f"Expected output file was not created: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _process_sprint_video(video, comparison_id, label):

    original_name = os.path.basename(
        video.filename or "sprint_video.mp4"
    )

    extension = os.path.splitext(original_name)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {extension}"
        )

    video_id = f"{comparison_id}_{label}"

    input_path = os.path.join(
        UPLOAD_DIR,
        f"{video_id}{extension}"
    )

    output_video = os.path.join(
        PROCESSED_DIR,
        f"{video_id}_sprint_pose.mp4"
    )

    output_csv = os.path.join(
        PROCESSED_DIR,
        f"{video_id}_sprint_angles.csv"
    )

    features_json = os.path.join(
        PROCESSED_DIR,
        f"{video_id}_sprint_features.json"
    )

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Existing pose detection pipeline
    analyze_video(
        input_path,
        output_video,
        output_csv
    )

    # Existing Sprint feature extraction
    analyze_sprint(
        output_csv,
        features_json
    )

    # Read extracted features
    features = _read_json(features_json)

    # IMPORTANT:
    # Existing assess_sprint() returns the assessment directly.
    assessment = assess_sprint(features)

    if not isinstance(assessment, dict):
        raise RuntimeError(
            "Sprint assessment did not return a valid dictionary."
        )

    return {
        "filename": original_name,
        "processed_video_url": (
            f"/processed/{os.path.basename(output_video)}"
        ),
        "features": features,
        "assessment": assessment
    }


@router.post("/sprint")
async def compare_sprint_videos(
    video_a: UploadFile = File(...),
    video_b: UploadFile = File(...),
    comparison_mode: str = Form("progress")
):

    if comparison_mode not in ("progress", "reference"):
        raise HTTPException(
            status_code=400,
            detail="comparison_mode must be 'progress' or 'reference'."
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    comparison_id = str(uuid.uuid4())

    try:
        result_a = _process_sprint_video(
            video_a,
            comparison_id,
            "video_a"
        )

        result_b = _process_sprint_video(
            video_b,
            comparison_id,
            "video_b"
        )

        comparison = build_sprint_comparison(
            result_a["features"],
            result_a["assessment"],
            result_b["features"],
            result_b["assessment"],
            comparison_mode
        )

        comparison["analysis_id"] = comparison_id

        comparison["video_a"].update({
            "filename": result_a["filename"],
            "processed_video_url": result_a["processed_video_url"]
        })

        comparison["video_b"].update({
            "filename": result_b["filename"],
            "processed_video_url": result_b["processed_video_url"]
        })

        # Save comparison in existing Analysis History
        save_analysis_record(
            analysis_id=comparison_id,
            activity="Sprint Comparison",
            filename=(
                f"{result_a['filename']} vs {result_b['filename']}"
            ),
            processed_video_url=result_a["processed_video_url"],
            assessment=comparison
        )

        return comparison

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Sprint comparison failed: {str(error)}"
        ) from error

    finally:
        await video_a.close()
        await video_b.close()
