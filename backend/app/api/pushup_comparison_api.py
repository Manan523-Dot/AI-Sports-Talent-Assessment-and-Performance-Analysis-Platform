
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.services.pushup_pose_detection import analyze_pushup_video
from backend.app.services.pushup_analysis import analyze_pushup
from backend.app.ml.pushup_assessment import assess_pushup
from backend.app.ml.pushup_comparison_engine import compare_pushup_results
from backend.app.database.save_analysis import save_analysis_record


router = APIRouter(prefix="/api/compare", tags=["Video Comparison"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _save_upload(upload: UploadFile, destination: str):
    extension = Path(upload.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video format for {upload.filename}. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    with open(destination, "wb") as output_file:
        shutil.copyfileobj(upload.file, output_file)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _process_pushup_video(input_path: str, work_dir: str, label: str):
    """
    Run one video through the existing Push-up pipeline.

    Returns the assessment and CSV path used by the comparison engine.
    """
    output_video = os.path.join(work_dir, f"{label}_processed.mp4")
    output_csv = os.path.join(work_dir, f"{label}_pose_data.csv")
    features_json = os.path.join(work_dir, f"{label}_features.json")
    assessment_json = os.path.join(work_dir, f"{label}_assessment.json")

    # Existing pose-detection pipeline.
    analyze_pushup_video(input_path, output_video, output_csv)

    if not os.path.exists(output_csv):
        raise RuntimeError(f"Push-up pose detection did not create {label} CSV.")

    # Existing Push-up movement analysis.
    analyze_pushup(output_csv, features_json)

    if not os.path.exists(features_json):
        raise RuntimeError(f"Push-up analysis did not create {label} features.")

    # Existing Push-up assessment pipeline.
    assessment_result = assess_pushup(features_json, assessment_json)

    if isinstance(assessment_result, dict):
        assessment = assessment_result
    elif os.path.exists(assessment_json):
        assessment = _load_json(assessment_json)
    else:
        raise RuntimeError(
            f"Push-up assessment did not return or save {label} assessment."
        )

    return {
        "assessment": assessment,
        "csv_path": output_csv,
        "processed_video_path": output_video,
    }


@router.post("/pushup")
async def compare_pushup_videos(
    video_a: UploadFile = File(...),
    video_b: UploadFile = File(...),
    comparison_mode: str = Form("progress"),
):
    """
    Compare two Push-up videos.

    comparison_mode:
      - progress: Video A is the earlier attempt and Video B is the latest attempt.
      - reference: Video A is the reference and Video B is the athlete attempt.
    """
    if comparison_mode not in {"progress", "reference"}:
        raise HTTPException(
            status_code=400,
            detail="comparison_mode must be 'progress' or 'reference'.",
        )

    if not video_a.filename or not video_b.filename:
        raise HTTPException(status_code=400, detail="Both video files are required.")

    analysis_id = str(uuid.uuid4())

    try:
        with tempfile.TemporaryDirectory(prefix="pushup_comparison_") as work_dir:
            input_a = os.path.join(work_dir, f"video_a{Path(video_a.filename).suffix.lower()}")
            input_b = os.path.join(work_dir, f"video_b{Path(video_b.filename).suffix.lower()}")

            _save_upload(video_a, input_a)
            _save_upload(video_b, input_b)

            # Process each video independently using the existing modules.
            result_a = _process_pushup_video(input_a, work_dir, "video_a")
            result_b = _process_pushup_video(input_b, work_dir, "video_b")

            # Compare assessments and normalized movement curves.
            comparison = compare_pushup_results(
                assessment_a=result_a["assessment"],
                assessment_b=result_b["assessment"],
                csv_path_a=result_a["csv_path"],
                csv_path_b=result_b["csv_path"],
                comparison_mode=comparison_mode,
                filename_a=video_a.filename,
                filename_b=video_b.filename,
            )

            response = {
                "success": True,
                "analysis_id": analysis_id,
                "exercise": "Push-up",
                "comparison_mode": comparison_mode,
                "video_a": {
                    "filename": video_a.filename,
                    "assessment": result_a["assessment"],
                },
                "video_b": {
                    "filename": video_b.filename,
                    "assessment": result_b["assessment"],
                },
                "comparison": comparison,
            }

            # Save the combined comparison in the existing analysis history.
            try:
                save_analysis_record(
                    analysis_id,
                    activity="Push-up Comparison",
                    filename=f"{video_a.filename} vs {video_b.filename}",
                    processed_video_url=None,
                    assessment=response,
                )
            except Exception as database_error:
                # Keep a successful comparison usable even if history saving fails.
                response["history_saved"] = False
                response["history_warning"] = str(database_error)
            else:
                response["history_saved"] = True

            return response

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Push-up comparison failed: {str(error)}",
        ) from error
    finally:
        await video_a.close()
        await video_b.close()
