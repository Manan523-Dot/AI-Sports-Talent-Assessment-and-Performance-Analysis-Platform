
import os
import shutil
import uuid

import pandas as pd

from fastapi import APIRouter, File, UploadFile

from backend.app.services.pose_detection import analyze_video
from backend.app.services.squat_analysis import analyze_squat
from backend.app.ml.squat_assessment import assess_squat

from backend.app.database.save_analysis import save_analysis_record


router = APIRouter(
    prefix="/api/analyze",
    tags=["Sports Analysis"]
)


# =============================================================
# HELPER: PREPARE KNEE ANGLE CHART DATA
# =============================================================

def prepare_knee_angle_data(csv_path):

    df = pd.read_csv(csv_path)

    required_columns = [
        "time_seconds",
        "left_knee_angle",
        "right_knee_angle"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing column in squat CSV: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Use whichever side is visible in each frame.
    df["average_knee_angle"] = df[
        ["left_knee_angle", "right_knee_angle"]
    ].mean(axis=1)

    chart_data = []

    for _, row in df.iterrows():

        if pd.isna(row["time_seconds"]):
            continue

        left_angle = row["left_knee_angle"]
        right_angle = row["right_knee_angle"]
        average_angle = row["average_knee_angle"]

        chart_data.append({

            "time_seconds": round(
                float(row["time_seconds"]), 3
            ),

            "left_knee_angle": (
                round(float(left_angle), 2)
                if pd.notna(left_angle)
                else None
            ),

            "right_knee_angle": (
                round(float(right_angle), 2)
                if pd.notna(right_angle)
                else None
            ),

            "average_knee_angle": (
                round(float(average_angle), 2)
                if pd.notna(average_angle)
                else None
            )
        })

    return chart_data


# =============================================================
# HELPER: MEASUREMENT VISIBILITY
# =============================================================

def get_measurement_visibility(csv_path):

    df = pd.read_csv(csv_path)

    columns = [
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_trunk_angle",
        "right_trunk_angle"
    ]

    for column in columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing visibility column: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    total_frames = len(df)

    counts = {
        column: int(df[column].notna().sum())
        for column in columns
    }

    left_available = (
        counts["left_knee_angle"] > 0
        and counts["left_hip_angle"] > 0
    )

    right_available = (
        counts["right_knee_angle"] > 0
        and counts["right_hip_angle"] > 0
    )

    if left_available and right_available:
        detected_sides = "both"

    elif left_available:
        detected_sides = "left"

    elif right_available:
        detected_sides = "right"

    else:
        detected_sides = "insufficient"

    paired_knee_frames = int(
        df[
            ["left_knee_angle", "right_knee_angle"]
        ].dropna().shape[0]
    )

    paired_hip_frames = int(
        df[
            ["left_hip_angle", "right_hip_angle"]
        ].dropna().shape[0]
    )

    return {

        "total_frames": total_frames,

        "detected_sides": detected_sides,

        "left_knee_frames": counts["left_knee_angle"],

        "right_knee_frames": counts["right_knee_angle"],

        "left_hip_frames": counts["left_hip_angle"],

        "right_hip_frames": counts["right_hip_angle"],

        "left_trunk_frames": counts["left_trunk_angle"],

        "right_trunk_frames": counts["right_trunk_angle"],

        "paired_knee_frames": paired_knee_frames,

        "paired_hip_frames": paired_hip_frames,

        "knee_symmetry_available": paired_knee_frames >= 5,

        "hip_symmetry_available": paired_hip_frames >= 5
    }


# =============================================================
# SQUAT VIDEO ANALYSIS
# =============================================================

@router.post("/squat")
async def analyze_squat_video(
    video: UploadFile = File(...)
):

    # ---------------------------------------------------------
    # CREATE UNIQUE ANALYSIS ID
    # ---------------------------------------------------------

    analysis_id = str(uuid.uuid4())

    # ---------------------------------------------------------
    # FILE PATHS
    # ---------------------------------------------------------

    input_video = (
        f"data/uploads/{analysis_id}.mp4"
    )

    output_video = (
        f"data/processed/{analysis_id}_pose.mp4"
    )

    output_csv = (
        f"data/processed/{analysis_id}_angles.csv"
    )

    output_features = (
        f"data/processed/{analysis_id}_features.json"
    )

    output_assessment = (
        f"data/processed/{analysis_id}_assessment.json"
    )

    # ---------------------------------------------------------
    # CREATE REQUIRED FOLDERS
    # ---------------------------------------------------------

    os.makedirs(
        "data/uploads",
        exist_ok=True
    )

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # ---------------------------------------------------------
    # SAVE UPLOADED VIDEO
    # ---------------------------------------------------------

    with open(input_video, "wb") as buffer:

        shutil.copyfileobj(
            video.file,
            buffer
        )

    # ---------------------------------------------------------
    # STEP 1: POSE DETECTION
    # ---------------------------------------------------------

    analyze_video(
        input_video,
        output_video,
        output_csv
    )

    # ---------------------------------------------------------
    # STEP 2: FEATURE EXTRACTION
    # ---------------------------------------------------------

    features = analyze_squat(
        output_csv,
        output_features
    )

    # ---------------------------------------------------------
    # STEP 3: PERFORMANCE ASSESSMENT
    # ---------------------------------------------------------

    assessment = assess_squat(
        output_features,
        output_assessment
    )

    # ---------------------------------------------------------
    # STEP 4: PREPARE CHART DATA
    # ---------------------------------------------------------

    knee_angle_data = prepare_knee_angle_data(
        output_csv
    )

    # ---------------------------------------------------------
    # STEP 5: MEASUREMENT VISIBILITY
    # ---------------------------------------------------------

    measurement_visibility = get_measurement_visibility(
        output_csv
    )

    # ---------------------------------------------------------
    # PROCESSED VIDEO URL
    # ---------------------------------------------------------

    processed_video_url = (
        f"/processed/{analysis_id}_pose.mp4"
    )

    # ---------------------------------------------------------
    # STEP 6: SAVE RESULT IN DATABASE
    # ---------------------------------------------------------

    save_analysis_record(
        analysis_id=analysis_id,
        activity="Squat",
        filename=video.filename,
        processed_video_url=processed_video_url,
        assessment=assessment
    )

    # ---------------------------------------------------------
    # FINAL RESPONSE
    # ---------------------------------------------------------

    return {

        "analysis_id": analysis_id,

        "filename": video.filename,

        "processed_video_url": processed_video_url,

        "assessment": assessment,

        "measurement_visibility": measurement_visibility,

        "knee_angle_data": knee_angle_data
    }
