
import os
import tempfile
import uuid

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from backend.app.services.lunge_pose_detection import (
    analyze_lunge_video,
)
from backend.app.services.lunge_analysis import (
    analyze_lunge,
)
from backend.app.ml.lunge_assessment import (
    assess_lunge,
)
from backend.app.ml.lunge_comparison_engine import (
    compare_lunge_results,
)
from backend.app.database.save_analysis import (
    save_analysis_record,
)


router = APIRouter(
    prefix="/api/compare",
    tags=["Video Comparison"],
)


PROCESSED_DIR = "data/processed"


async def save_upload(upload: UploadFile, path: str):
    content = await upload.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded video is empty: {upload.filename}",
        )

    with open(path, "wb") as file:
        file.write(content)


def process_lunge_video(
    video_path,
    work_dir,
    label,
    permanent_video_path,
):
    """
    Process one Lunge video.

    Intermediate analysis files remain temporary.
    The processed video is saved permanently.
    """

    csv_path = os.path.join(
        work_dir,
        f"{label}_features.csv",
    )

    analysis_path = os.path.join(
        work_dir,
        f"{label}_analysis.json",
    )

    assessment_path = os.path.join(
        work_dir,
        f"{label}_assessment.json",
    )

    # Generate the processed video and pose CSV.
    analyze_lunge_video(
        video_path,
        permanent_video_path,
        csv_path,
    )

    # Extract Lunge features.
    analysis = analyze_lunge(
        csv_path,
        analysis_path,
    )

    # Generate performance assessment.
    assessment = assess_lunge(
        analysis_path,
        assessment_path,
    )

    if not os.path.exists(permanent_video_path):
        raise RuntimeError(
            f"Processed video was not generated for {label}."
        )

    return assessment, csv_path


@router.post("/lunge")
async def compare_lunge_endpoint(
    video_a: UploadFile = File(...),
    video_b: UploadFile = File(...),
    comparison_mode: str = Form("progress"),
):
    if comparison_mode not in ["progress", "reference"]:
        raise HTTPException(
            status_code=400,
            detail="comparison_mode must be progress or reference.",
        )

    comparison_id = str(uuid.uuid4())

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True,
    )

    # Unique permanent video filenames.
    filename_a = f"lunge_{comparison_id}_video_a.mp4"
    filename_b = f"lunge_{comparison_id}_video_b.mp4"

    permanent_video_a = os.path.join(
        PROCESSED_DIR,
        filename_a,
    )

    permanent_video_b = os.path.join(
        PROCESSED_DIR,
        filename_b,
    )

    video_url_a = f"/processed/{filename_a}"
    video_url_b = f"/processed/{filename_b}"

    try:
        with tempfile.TemporaryDirectory(
            prefix="lunge_comparison_"
        ) as work_dir:

            # Save uploaded source videos temporarily.
            input_a = os.path.join(
                work_dir,
                "video_a.mp4",
            )

            input_b = os.path.join(
                work_dir,
                "video_b.mp4",
            )

            await save_upload(video_a, input_a)
            await save_upload(video_b, input_b)

            # Process both videos independently.
            assessment_a, csv_path_a = process_lunge_video(
                input_a,
                work_dir,
                "video_a",
                permanent_video_a,
            )

            assessment_b, csv_path_b = process_lunge_video(
                input_b,
                work_dir,
                "video_b",
                permanent_video_b,
            )

            # Compare the two assessments.
            comparison = compare_lunge_results(
                assessment_a,
                assessment_b,
                csv_path_a,
                csv_path_b,
                comparison_mode,
            )

        # Temporary files are deleted here.
        # Permanent processed videos remain available.

        # Attach processed video URLs to the response.
        comparison["video_a"]["processed_video_url"] = (
            video_url_a
        )

        comparison["video_a"]["video_url"] = video_url_a

        comparison["video_b"]["processed_video_url"] = (
            video_url_b
        )

        comparison["video_b"]["video_url"] = video_url_b

        comparison["comparison_id"] = comparison_id
        comparison["history_saved"] = False

        # Save comparison history.
        save_analysis_record(
            analysis_id=comparison_id,
            activity="Lunge Comparison",
            filename=(
                f"{video_a.filename or 'Video A'} vs "
                f"{video_b.filename or 'Video B'}"
            ),
            processed_video_url=None,
            assessment=comparison,
        )

        comparison["history_saved"] = True

        return comparison

    except HTTPException:
        # Remove partially generated videos on failure.
        for path in [permanent_video_a, permanent_video_b]:
            if os.path.exists(path):
                os.remove(path)

        raise

    except Exception as error:
        # Remove partially generated videos on failure.
        for path in [permanent_video_a, permanent_video_b]:
            if os.path.exists(path):
                os.remove(path)

        raise HTTPException(
            status_code=500,
            detail=f"Lunge comparison failed: {str(error)}",
        )
