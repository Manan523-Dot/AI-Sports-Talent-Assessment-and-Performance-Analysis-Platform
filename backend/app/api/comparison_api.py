
import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.services.pose_detection import analyze_video
from backend.app.services.squat_analysis import analyze_squat
from backend.app.ml.squat_assessment import assess_squat

from backend.app.ml.comparison_engine import (
    extract_normalized_knee_curve,
    build_squat_comparison,
)

from backend.app.ml.comparison_report import (
    build_squat_comparison_report,
)

from backend.app.database.save_analysis import save_analysis_record


router = APIRouter(
    prefix="/api/compare",
    tags=["Video Comparison"]
)

BASE_DIR = Path(__file__).resolve().parents[3]

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def process_squat_video(upload: UploadFile, label: str):
    """
    Run the existing Squat pipeline without modifying its modules.
    """

    extension = Path(upload.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format for {label}."
        )

    video_id = str(uuid.uuid4())

    input_path = UPLOAD_DIR / f"{video_id}{extension}"
    processed_path = PROCESSED_DIR / f"{video_id}_pose.mp4"
    angles_path = PROCESSED_DIR / f"{video_id}_angles.csv"
    features_path = PROCESSED_DIR / f"{video_id}_features.json"
    assessment_path = PROCESSED_DIR / f"{video_id}_assessment.json"

    created_files = [
        input_path,
        processed_path,
        angles_path,
        features_path,
        assessment_path
    ]

    try:
        upload.file.seek(0)

        with open(input_path, "wb") as destination:
            shutil.copyfileobj(upload.file, destination)

        # Existing pose detection
        analyze_video(
            str(input_path),
            str(processed_path),
            str(angles_path)
        )

        # Existing Squat analysis
        analyze_squat(
            str(angles_path),
            str(features_path)
        )

        # Existing Squat assessment
        assessment = assess_squat(
            str(features_path),
            str(assessment_path)
        )

        if assessment is None and assessment_path.exists():
            with open(
                assessment_path,
                "r",
                encoding="utf-8"
            ) as file:
                assessment = json.load(file)

        if not isinstance(assessment, dict):
            raise ValueError(
                f"Squat assessment failed for {label}."
            )

        # Existing knee-angle curve extraction
        knee_curve = extract_normalized_knee_curve(
            str(angles_path)
        )

        return {
            "video_id": video_id,
            "filename": upload.filename,
            "processed_video_url": (
                f"/processed/{video_id}_pose.mp4"
            ),
            "assessment": assessment,
            "knee_angle_curve": knee_curve,
            "created_files": created_files
        }

    except HTTPException:
        cleanup_comparison_files(created_files)
        raise

    except Exception as error:
        cleanup_comparison_files(created_files)

        raise RuntimeError(
            f"Failed to process {label}: {str(error)}"
        ) from error


def cleanup_comparison_files(file_paths):
    """Delete generated files when comparison processing fails."""

    for file_path in file_paths:
        try:
            path = Path(file_path)

            if path.exists() and path.is_file():
                path.unlink()

        except OSError:
            pass


def build_comparison_report(
    assessment_a,
    assessment_b,
    curve_a,
    curve_b,
    comparison_mode
):
    """
    Build the athlete-facing report using the isolated report module.
    """

    report = build_squat_comparison_report(
        assessment_a=assessment_a,
        assessment_b=assessment_b,
        curve_a=curve_a,
        curve_b=curve_b,
        comparison_mode=comparison_mode
    )

    if not isinstance(report, dict):
        raise ValueError(
            "Comparison report generation returned invalid data."
        )

    report_insights = report.get("insights")

    if not isinstance(report_insights, dict):
        raise ValueError(
            "Comparison report does not contain valid insights."
        )

    return report


@router.post("/squat")
async def compare_squat_videos(
    video_a: UploadFile = File(...),
    video_b: UploadFile = File(...),
    comparison_mode: str = Form("progress")
):
    """
    Compare two Squat videos.

    Modes:
    - progress: older video versus newer video
    - reference: athlete video versus reference video
    """

    comparison_mode = comparison_mode.strip().lower()

    if comparison_mode not in {"progress", "reference"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "comparison_mode must be 'progress' or 'reference'."
            )
        )

    created_files = []

    try:
        if not video_a.filename or not video_b.filename:
            raise HTTPException(
                status_code=400,
                detail="Both video files are required."
            )

        # -------------------------------------------------
        # 1. Process both videos using existing Squat system
        # -------------------------------------------------

        result_a = process_squat_video(
            video_a,
            "Video A"
        )

        created_files.extend(result_a["created_files"])

        result_b = process_squat_video(
            video_b,
            "Video B"
        )

        created_files.extend(result_b["created_files"])

        assessment_a = result_a["assessment"]
        assessment_b = result_b["assessment"]

        curve_a = result_a["knee_angle_curve"]
        curve_b = result_b["knee_angle_curve"]

        # -------------------------------------------------
        # 2. Existing comparison engine
        # -------------------------------------------------

        comparison = build_squat_comparison(
            assessment_a,
            assessment_b,
            curve_a,
            curve_b,
            comparison_mode
        )

        if not isinstance(comparison, dict):
            comparison = {
                "engine_result": comparison
            }

        # -------------------------------------------------
        # 3. Generate refined comparison report
        # -------------------------------------------------

        report = build_comparison_report(
            assessment_a=assessment_a,
            assessment_b=assessment_b,
            curve_a=curve_a,
            curve_b=curve_b,
            comparison_mode=comparison_mode
        )

        report_insights = report["insights"]

        # Use filtered athlete-facing metrics instead of
        # exposing technical fields from the old engine.
        comparison["metrics"] = report["metrics"]

        # Keep the engine's other results while attaching
        # the refined report.
        comparison["report"] = report
        comparison["insights"] = report_insights

        # -------------------------------------------------
        # 4. Combine limitations without duplicates
        # -------------------------------------------------

        limitations = list(
            report_insights.get("limitations", [])
        )

        engine_limitations = comparison.get("limitations", [])

        if isinstance(engine_limitations, list):
            limitations.extend(engine_limitations)

        limitations = list(dict.fromkeys(limitations))

        # -------------------------------------------------
        # 5. Prepare comparison history record
        # -------------------------------------------------

        comparison_id = str(uuid.uuid4())

        history_assessment = {
            "comparison_id": comparison_id,
            "comparison_mode": comparison_mode,
            "exercise": "Squat",

            "video_a": {
                "filename": result_a["filename"],
                "processed_video_url": (
                    result_a["processed_video_url"]
                ),
                "assessment": assessment_a
            },

            "video_b": {
                "filename": result_b["filename"],
                "processed_video_url": (
                    result_b["processed_video_url"]
                ),
                "assessment": assessment_b
            },

            "comparison": comparison,
            "insights": report_insights,
            "report": report,
            "limitations": limitations
        }

        # -------------------------------------------------
        # 6. Save using existing Analysis History database
        # -------------------------------------------------

        save_analysis_record(
            analysis_id=comparison_id,
            activity="Squat Comparison",
            filename=(
                f"{result_a['filename']} vs "
                f"{result_b['filename']}"
            ),
            processed_video_url=None,
            assessment=history_assessment
        )

        # -------------------------------------------------
        # 7. Return API response
        # -------------------------------------------------

        return {
            "success": True,
            "comparison_id": comparison_id,
            "exercise": "Squat",
            "comparison_mode": comparison_mode,

            "video_a": {
                "filename": result_a["filename"],
                "processed_video_url": (
                    result_a["processed_video_url"]
                ),
                "assessment": assessment_a,
                "knee_angle_curve": curve_a
            },

            "video_b": {
                "filename": result_b["filename"],
                "processed_video_url": (
                    result_b["processed_video_url"]
                ),
                "assessment": assessment_b,
                "knee_angle_curve": curve_b
            },

            "comparison": comparison,
            "insights": report_insights,
            "report": report,
            "limitations": limitations
        }

    except HTTPException:
        cleanup_comparison_files(created_files)
        raise

    except Exception as error:
        cleanup_comparison_files(created_files)

        raise HTTPException(
            status_code=500,
            detail=f"Squat comparison failed: {str(error)}"
        ) from error

    finally:
        await video_a.close()
        await video_b.close()
