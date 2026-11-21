
import os
import shutil
import uuid

import pandas as pd

from fastapi import APIRouter, File, UploadFile

from backend.app.services.pushup_pose_detection import analyze_pushup_video
from backend.app.services.pushup_analysis import analyze_pushup
from backend.app.ml.pushup_assessment import assess_pushup

from backend.app.database.save_analysis import save_analysis_record


router = APIRouter(
    prefix="/api/analyze",
    tags=["Sports Analysis"]
)


# =============================================================
# HELPER: PREPARE ELBOW ANGLE CHART DATA
# =============================================================

def prepare_elbow_angle_data(csv_path):

    df = pd.read_csv(csv_path)

    required_columns = [
        "time_seconds",
        "left_elbow_angle",
        "right_elbow_angle"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing column in Push-up CSV: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["average_elbow_angle"] = df[
        ["left_elbow_angle", "right_elbow_angle"]
    ].mean(axis=1)

    chart_data = []

    for _, row in df.iterrows():

        if pd.isna(row["time_seconds"]):
            continue

        left_angle = row["left_elbow_angle"]
        right_angle = row["right_elbow_angle"]
        average_angle = row["average_elbow_angle"]

        chart_data.append({
            "time_seconds": round(
                float(row["time_seconds"]), 3
            ),

            "left_elbow_angle": (
                round(float(left_angle), 2)
                if pd.notna(left_angle)
                else None
            ),

            "right_elbow_angle": (
                round(float(right_angle), 2)
                if pd.notna(right_angle)
                else None
            ),

            "average_elbow_angle": (
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

    visibility_columns = [
        "left_elbow_visibility",
        "right_elbow_visibility",
        "left_body_visibility",
        "right_body_visibility"
    ]

    angle_columns = [
        "left_elbow_angle",
        "right_elbow_angle",
        "left_body_alignment",
        "right_body_alignment"
    ]

    required_columns = visibility_columns + angle_columns

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing visibility column in Push-up CSV: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    total_frames = len(df)

    counts = {
        column: int(df[column].fillna(0).gt(0).sum())
        for column in visibility_columns
    }

    angle_counts = {
        column: int(df[column].notna().sum())
        for column in angle_columns
    }

    left_available = (
        angle_counts["left_elbow_angle"] > 0
        and angle_counts["left_body_alignment"] > 0
    )

    right_available = (
        angle_counts["right_elbow_angle"] > 0
        and angle_counts["right_body_alignment"] > 0
    )

    if left_available and right_available:
        detected_sides = "both"
    elif left_available:
        detected_sides = "left"
    elif right_available:
        detected_sides = "right"
    else:
        detected_sides = "insufficient"

    paired_elbow_frames = int(
        df[
            ["left_elbow_angle", "right_elbow_angle"]
        ].dropna().shape[0]
    )

    paired_body_frames = int(
        df[
            ["left_body_alignment", "right_body_alignment"]
        ].dropna().shape[0]
    )

    return {
        "total_frames": total_frames,

        "detected_sides": detected_sides,

        "left_elbow_frames": angle_counts["left_elbow_angle"],
        "right_elbow_frames": angle_counts["right_elbow_angle"],

        "left_body_frames": angle_counts["left_body_alignment"],
        "right_body_frames": angle_counts["right_body_alignment"],

        "left_elbow_visible_frames": counts["left_elbow_visibility"],
        "right_elbow_visible_frames": counts["right_elbow_visibility"],

        "left_body_visible_frames": counts["left_body_visibility"],
        "right_body_visible_frames": counts["right_body_visibility"],

        "paired_elbow_frames": paired_elbow_frames,
        "paired_body_frames": paired_body_frames,

        "elbow_symmetry_available": paired_elbow_frames >= 5,
        "body_alignment_symmetry_available": paired_body_frames >= 5
    }


# =============================================================
# PUSH-UP VIDEO ANALYSIS
# =============================================================

@router.post("/pushup")
async def analyze_pushup_video_endpoint(
    video: UploadFile = File(...)
):

    # ---------------------------------------------------------
    # CREATE UNIQUE ANALYSIS ID
    # ---------------------------------------------------------

    analysis_id = str(uuid.uuid4())

    # ---------------------------------------------------------
    # FILE PATHS
    # ---------------------------------------------------------

    input_video = f"data/uploads/{analysis_id}.mp4"

    output_video = f"data/processed/{analysis_id}_pose.mp4"

    output_csv = f"data/processed/{analysis_id}_features.csv"

    output_features = f"data/processed/{analysis_id}_features.json"

    output_assessment = f"data/processed/{analysis_id}_assessment.json"

    # ---------------------------------------------------------
    # CREATE REQUIRED FOLDERS
    # ---------------------------------------------------------

    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

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

    analyze_pushup_video(
        input_video,
        output_video,
        output_csv
    )

    # ---------------------------------------------------------
    # STEP 2: FEATURE EXTRACTION
    # ---------------------------------------------------------

    features = analyze_pushup(
        output_csv,
        output_features
    )

    # ---------------------------------------------------------
    # STEP 3: PERFORMANCE ASSESSMENT
    # ---------------------------------------------------------

    assessment = assess_pushup(
        output_features,
        output_assessment
    )

    # ---------------------------------------------------------
    # STEP 4: PREPARE CHART DATA
    # ---------------------------------------------------------

    elbow_angle_data = prepare_elbow_angle_data(
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
        activity="Push-up",
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

        "elbow_angle_data": elbow_angle_data
    }
