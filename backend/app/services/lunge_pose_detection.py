
import csv
import math
import os
import subprocess

import cv2
import mediapipe as mp

from imageio_ffmpeg import get_ffmpeg_exe


MODEL_PATH = "models/pose/pose_landmarker_full.task"
MIN_VISIBILITY = 0.5

LEFT_SHOULDER = 11
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

RIGHT_SHOULDER = 12
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28

CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]


def calculate_angle(a, b, c):
    if a is None or b is None or c is None:
        return None

    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2
    )
    mag_bc = math.sqrt( bc[0] ** 2 + bc[1] ** 2
    )

    if mag_ba == 0 or mag_bc == 0:
        return None

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    cosine = dot / (mag_ba * mag_bc)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def is_visible(landmark):
    return (
        landmark is not None
        and landmark.visibility >= MIN_VISIBILITY
    )


def get_point(landmarks, index):
    landmark = landmarks[index]

    if not is_visible(landmark):
        return None

    return (landmark.x, landmark.y)


def get_angle(landmarks, first, middle, last):
    return calculate_angle(
        get_point(landmarks, first),
        get_point(landmarks, middle),
        get_point(landmarks, last)
    )


def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]

    for first, second in CONNECTIONS:
        point_a = get_point(landmarks, first)
        point_b = get_point(landmarks, second)

        if point_a is None or point_b is None:
            continue

        cv2.line(
            frame,
            (int(point_a[0] * width), int(point_a[1] * height)),
            (int(point_b[0] * width), int(point_b[1] * height)),
            (0, 255, 0),
            2
        )

    for index in set(
        index for connection in CONNECTIONS for index in connection
    ):
        point = get_point(landmarks, index)

        if point is not None:
            cv2.circle(
                frame,
                (int(point[0] * width), int(point[1] * height)),
                4,
                (0, 0, 255),
                -1
            )


def convert_to_browser_video(input_path, output_path):
    ffmpeg = get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
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


def analyze_lunge_video(input_video, output_video, output_csv):

    if not os.path.exists(input_video):
        raise FileNotFoundError(
            f"Input video not found: {input_video}"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found: {MODEL_PATH}"
        )

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    temp_video = output_video.replace(".mp4", "_temp.mp4")

    writer = cv2.VideoWriter(
        temp_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create processed video.")

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

    fieldnames = [
        "frame",
        "time_seconds",
        "left_knee_angle",
        "right_knee_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_torso_angle",
        "right_torso_angle",
        "left_knee_visibility",
        "right_knee_visibility",
        "left_hip_visibility",
        "right_hip_visibility"
    ]

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

                    rgb_frame = cv2.cvtColor(
                        frame,
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

                    left_knee = None
                    right_knee = None
                    left_hip = None
                    right_hip = None
                    left_torso = None
                    right_torso = None

                    left_knee_visibility = 0
                    right_knee_visibility = 0
                    left_hip_visibility = 0
                    right_hip_visibility = 0

                    if result.pose_landmarks:

                        landmarks = result.pose_landmarks[0]
                        draw_pose(frame, landmarks)

                        left_knee = get_angle(
                            landmarks,
                            LEFT_HIP,
                            LEFT_KNEE,
                            LEFT_ANKLE
                        )

                        right_knee = get_angle(
                            landmarks,
                            RIGHT_HIP,
                            RIGHT_KNEE,
                            RIGHT_ANKLE
                        )

                        left_hip = get_angle(
                            landmarks,
                            LEFT_SHOULDER,
                            LEFT_HIP,
                            LEFT_KNEE
                        )

                        right_hip = get_angle(
                            landmarks,
                            RIGHT_SHOULDER,
                            RIGHT_HIP,
                            RIGHT_KNEE
                        )

                        left_torso = get_angle(
                            landmarks,
                            LEFT_SHOULDER,
                            LEFT_HIP,
                            LEFT_ANKLE
                        )

                        right_torso = get_angle(
                            landmarks,
                            RIGHT_SHOULDER,
                            RIGHT_HIP,
                            RIGHT_ANKLE
                        )

                        left_knee_visibility = int(
                            all(
                                is_visible(landmarks[i])
                                for i in [
                                    LEFT_HIP,
                                    LEFT_KNEE,
                                    LEFT_ANKLE
                                ]
                            )
                        )

                        right_knee_visibility = int(
                            all(
                                is_visible(landmarks[i])
                                for i in [
                                    RIGHT_HIP,
                                    RIGHT_KNEE,
                                    RIGHT_ANKLE
                                ]
                            )
                        )

                        left_hip_visibility = int(
                            all(
                                is_visible(landmarks[i])
                                for i in [
                                    LEFT_SHOULDER,
                                    LEFT_HIP,
                                    LEFT_KNEE
                                ]
                            )
                        )

                        right_hip_visibility = int(
                            all(
                                is_visible(landmarks[i])
                                for i in [
                                    RIGHT_SHOULDER,
                                    RIGHT_HIP,
                                    RIGHT_KNEE
                                ]
                            )
                        )

                    csv_writer.writerow({
                        "frame": frame_number,
                        "time_seconds": round(frame_number / fps, 3),
                        "left_knee_angle": left_knee,
                        "right_knee_angle": right_knee,
                        "left_hip_angle": left_hip,
                        "right_hip_angle": right_hip,
                        "left_torso_angle": left_torso,
                        "right_torso_angle": right_torso,
                        "left_knee_visibility": left_knee_visibility,
                        "right_knee_visibility": right_knee_visibility,
                        "left_hip_visibility": left_hip_visibility,
                        "right_hip_visibility": right_hip_visibility
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

    finally:
        cap.release()
        writer.release()

    if frame_number == 0:
        raise ValueError("No frames were read from the video.")

    convert_to_browser_video(
        temp_video,
        output_video
    )

    print("\nLUNGE POSE DETECTION COMPLETED")
    print("Frames processed:", frame_number)
    print("CSV saved:", output_csv)
    print("Processed video saved:", output_video)

    return output_csv
