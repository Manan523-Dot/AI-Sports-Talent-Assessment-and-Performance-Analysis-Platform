
import csv
import math
import os
import subprocess
import time
from pathlib import Path

import cv2
import mediapipe as mp

from imageio_ffmpeg import get_ffmpeg_exe


# =============================================================
# CONFIGURATION
# =============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = str(
    PROJECT_DIR / "models" / "pose" / "pose_landmarker_full.task"
)

MIN_VISIBILITY = 0.5

# Resize only the inference frame to reduce MediaPipe processing time.
# The final annotated video retains its original resolution.
MAX_INFERENCE_DIMENSION = 640

PROGRESS_INTERVAL = 50


# =============================================================
# MEDIAPIPE LANDMARK INDICES
# =============================================================

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
LEFT_HIP = 23
LEFT_ANKLE = 27

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
RIGHT_HIP = 24
RIGHT_ANKLE = 28


CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12), (11, 23), (12, 24),
    (23, 24), (23, 27), (24, 28)
]


# =============================================================
# ANGLE CALCULATION
# =============================================================

def calculate_angle(a, b, c):

    if a is None or b is None or c is None:
        return None

    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    magnitude_ba = math.sqrt(
        ba[0] ** 2 + ba[1] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 + bc[1] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return None

    dot_product = (
        ba[0] * bc[0] +
        ba[1] * bc[1]
    )

    cosine = dot_product / (
        magnitude_ba * magnitude_bc
    )

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


# =============================================================
# LANDMARK HELPERS
# =============================================================

def is_visible(landmark):

    return (
        landmark is not None
        and landmark.visibility >= MIN_VISIBILITY
    )


def get_point(landmarks, index):

    landmark = landmarks[index]

    if not is_visible(landmark):
        return None

    return (
        landmark.x,
        landmark.y
    )


def get_angle(landmarks, first, middle, last):

    a = get_point(landmarks, first)
    b = get_point(landmarks, middle)
    c = get_point(landmarks, last)

    return calculate_angle(a, b, c)


def get_body_alignment(landmarks, shoulder, hip, ankle):

    return get_angle(
        landmarks,
        shoulder,
        hip,
        ankle
    )


# =============================================================
# POSE ANNOTATION
# =============================================================

def draw_pose(frame, landmarks):

    height, width = frame.shape[:2]

    for first, second in CONNECTIONS:

        point_a = get_point(landmarks, first)
        point_b = get_point(landmarks, second)

        if point_a is None or point_b is None:
            continue

        x1 = int(point_a[0] * width)
        y1 = int(point_a[1] * height)

        x2 = int(point_b[0] * width)
        y2 = int(point_b[1] * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    landmark_indices = {
        index
        for connection in CONNECTIONS
        for index in connection
    }

    for index in landmark_indices:

        point = get_point(landmarks, index)

        if point is None:
            continue

        x = int(point[0] * width)
        y = int(point[1] * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


# =============================================================
# RESIZE FRAME FOR FASTER POSE DETECTION
# =============================================================

def prepare_inference_frame(frame):

    height, width = frame.shape[:2]

    largest_dimension = max(width, height)

    if largest_dimension <= MAX_INFERENCE_DIMENSION:
        return frame

    scale = MAX_INFERENCE_DIMENSION / largest_dimension

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# =============================================================
# BROWSER VIDEO CONVERSION
# =============================================================

def convert_to_browser_video(input_path, output_path):

    ffmpeg = get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        output_path
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Video conversion failed: {result.stderr[-2000:]}"
        )

    if os.path.exists(input_path):
        os.remove(input_path)


# =============================================================
# PUSH-UP POSE DETECTION
# =============================================================

def analyze_pushup_video(
    input_video,
    output_video,
    output_csv
):

    start_time = time.time()

    if not os.path.exists(input_video):

        raise FileNotFoundError(
            f"Input video not found: {input_video}"
        )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Pose model not found: {MODEL_PATH}"
        )

    os.makedirs(
        os.path.dirname(output_video) or ".",
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(output_csv) or ".",
        exist_ok=True
    )

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():

        raise ValueError(
            "Could not open uploaded video."
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    temp_video = output_video.replace(
        ".mp4",
        "_temp.mp4"
    )

    writer = cv2.VideoWriter(
        temp_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create processed video."
        )

    options = mp.tasks.vision.PoseLandmarkerOptions(

        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=mp.tasks.vision.RunningMode.VIDEO,

        num_poses=1,

        min_pose_detection_confidence=0.5,

        min_pose_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )

    frame_number = 0

    try:

        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            with open(
                output_csv,
                "w",
                newline="",
                encoding="utf-8"
            ) as csv_file:

                fieldnames = [
                    "frame",
                    "time_seconds",
                    "left_elbow_angle",
                    "right_elbow_angle",
                    "left_body_alignment",
                    "right_body_alignment",
                    "left_elbow_visibility",
                    "right_elbow_visibility",
                    "left_body_visibility",
                    "right_body_visibility"
                ]

                csv_writer = csv.DictWriter(
                    csv_file,
                    fieldnames=fieldnames
                )

                csv_writer.writeheader()

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    timestamp_ms = int(
                        (frame_number / fps) * 1000
                    )

                    # Resize only the inference frame.
                    inference_frame = prepare_inference_frame(
                        frame
                    )

                    rgb_frame = cv2.cvtColor(
                        inference_frame,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    result = landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )

                    left_elbow_angle = None
                    right_elbow_angle = None

                    left_alignment = None
                    right_alignment = None

                    left_elbow_visibility = 0
                    right_elbow_visibility = 0

                    left_body_visibility = 0
                    right_body_visibility = 0

                    if result.pose_landmarks:

                        landmarks = result.pose_landmarks[0]

                        # Coordinates are normalized, so the
                        # detected pose can be drawn on the
                        # original-resolution frame.

                        draw_pose(
                            frame,
                            landmarks
                        )

                        left_elbow_angle = get_angle(
                            landmarks,
                            LEFT_SHOULDER,
                            LEFT_ELBOW,
                            LEFT_WRIST
                        )

                        right_elbow_angle = get_angle(
                            landmarks,
                            RIGHT_SHOULDER,
                            RIGHT_ELBOW,
                            RIGHT_WRIST
                        )

                        left_alignment = get_body_alignment(
                            landmarks,
                            LEFT_SHOULDER,
                            LEFT_HIP,
                            LEFT_ANKLE
                        )

                        right_alignment = get_body_alignment(
                            landmarks,
                            RIGHT_SHOULDER,
                            RIGHT_HIP,
                            RIGHT_ANKLE
                        )

                        left_elbow_visibility = int(
                            all(
                                is_visible(landmarks[index])
                                for index in [
                                    LEFT_SHOULDER,
                                    LEFT_ELBOW,
                                    LEFT_WRIST
                                ]
                            )
                        )

                        right_elbow_visibility = int(
                            all(
                                is_visible(landmarks[index])
                                for index in [
                                    RIGHT_SHOULDER,
                                    RIGHT_ELBOW,
                                    RIGHT_WRIST
                                ]
                            )
                        )

                        left_body_visibility = int(
                            all(
                                is_visible(landmarks[index])
                                for index in [
                                    LEFT_SHOULDER,
                                    LEFT_HIP,
                                    LEFT_ANKLE
                                ]
                            )
                        )

                        right_body_visibility = int(
                            all(
                                is_visible(landmarks[index])
                                for index in [
                                    RIGHT_SHOULDER,
                                    RIGHT_HIP,
                                    RIGHT_ANKLE
                                ]
                            )
                        )

                    csv_writer.writerow({

                        "frame": frame_number,

                        "time_seconds": round(
                            frame_number / fps,
                            3
                        ),

                        "left_elbow_angle": left_elbow_angle,

                        "right_elbow_angle": right_elbow_angle,

                        "left_body_alignment": left_alignment,

                        "right_body_alignment": right_alignment,

                        "left_elbow_visibility": left_elbow_visibility,

                        "right_elbow_visibility": right_elbow_visibility,

                        "left_body_visibility": left_body_visibility,

                        "right_body_visibility": right_body_visibility
                    })

                    cv2.putText(
                        frame,
                        f"Frame: {frame_number}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    writer.write(frame)

                    frame_number += 1

                    if frame_number % PROGRESS_INTERVAL == 0:

                        print(
                            f"Push-up processing: "
                            f"{frame_number}/{total_frames} frames",
                            flush=True
                        )

    finally:

        cap.release()
        writer.release()

    if frame_number == 0:

        raise ValueError(
            "No frames were read from the video."
        )

    print(
        "\nPose detection completed. Converting video...",
        flush=True
    )

    convert_to_browser_video(
        temp_video,
        output_video
    )

    elapsed_time = round(
        time.time() - start_time,
        2
    )

    print("\nPUSH-UP POSE DETECTION COMPLETED")

    print("Frames processed:", frame_number)

    print("CSV saved:", output_csv)

    print("Processed video saved:", output_video)

    print("Total processing time:", elapsed_time, "seconds")

    return output_csv


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    input_video = str(
        PROJECT_DIR / "data" / "samples" / "pushup_test.mp4"
    )

    output_video = str(
        PROJECT_DIR / "data" / "processed" / "pushup_pose_output.mp4"
    )

    output_csv = str(
        PROJECT_DIR / "data" / "processed" / "pushup_features.csv"
    )

    analyze_pushup_video(
        input_video,
        output_video,
        output_csv
    )
