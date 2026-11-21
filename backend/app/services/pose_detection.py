
import cv2
import mediapipe as mp
import csv
import os
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe

from backend.app.utils.angle_calculator import calculate_angle


# Minimum landmark visibility required for angle calculation.
MIN_VISIBILITY = 0.5


def convert_to_browser_video(input_video, output_video):
    """Convert OpenCV video to browser-compatible H.264 MP4."""

    ffmpeg_exe = get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_video,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_video,
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if os.path.exists(input_video):
        os.remove(input_video)


def is_landmark_visible(landmarks, indices):
    """Check whether all required landmarks have sufficient visibility."""

    return all(
        landmarks[index].visibility >= MIN_VISIBILITY
        for index in indices
    )


def analyze_video(input_video, output_video, output_csv):
    """
    Process sports video using MediaPipe Pose.

    Preserves the existing CSV structure and output behavior.
    Also records visibility diagnostics for left/right measurements.
    """

    model_path = "models/pose/pose_landmarker_full.task"

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open input video: {input_video}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    os.makedirs(
        os.path.dirname(output_video) or ".",
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(output_csv) or ".",
        exist_ok=True
    )

    temporary_video = output_video.replace(
        ".mp4",
        "_opencv.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        temporary_video,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise ValueError(
            "Could not create temporary output video."
        )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    connections = [
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
        (27, 29), (29, 31),
        (28, 30), (30, 32),
    ]

    angle_data = []
    frame_number = 0

    # Diagnostic counters.
    visibility_counts = {
        "left_knee": 0,
        "right_knee": 0,
        "both_knees": 0,
        "left_hip": 0,
        "right_hip": 0,
        "both_hips": 0,
    }

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(
                    frame_number * 1000 / fps
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                left_knee_angle = None
                right_knee_angle = None

                left_hip_angle = None
                right_hip_angle = None

                left_trunk_angle = None
                right_trunk_angle = None

                pelvis_x = None
                pelvis_y = None
                pelvis_visibility = None

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]

                    # Left knee: hip, knee, ankle.
                    if is_landmark_visible(
                        landmarks, [23, 25, 27]
                    ):
                        left_knee_angle = calculate_angle(
                            [landmarks[23].x, landmarks[23].y],
                            [landmarks[25].x, landmarks[25].y],
                            [landmarks[27].x, landmarks[27].y]
                        )

                    # Right knee: hip, knee, ankle.
                    if is_landmark_visible(
                        landmarks, [24, 26, 28]
                    ):
                        right_knee_angle = calculate_angle(
                            [landmarks[24].x, landmarks[24].y],
                            [landmarks[26].x, landmarks[26].y],
                            [landmarks[28].x, landmarks[28].y]
                        )

                    # Left hip: shoulder, hip, knee.
                    if is_landmark_visible(
                        landmarks, [11, 23, 25]
                    ):
                        left_hip_angle = calculate_angle(
                            [landmarks[11].x, landmarks[11].y],
                            [landmarks[23].x, landmarks[23].y],
                            [landmarks[25].x, landmarks[25].y]
                        )

                    # Right hip: shoulder, hip, knee.
                    if is_landmark_visible(
                        landmarks, [12, 24, 26]
                    ):
                        right_hip_angle = calculate_angle(
                            [landmarks[12].x, landmarks[12].y],
                            [landmarks[24].x, landmarks[24].y],
                            [landmarks[26].x, landmarks[26].y]
                        )

                    # Left trunk.
                    if is_landmark_visible(
                        landmarks, [11, 23]
                    ):
                        left_trunk_angle = calculate_angle(
                            [landmarks[11].x, landmarks[11].y],
                            [landmarks[23].x, landmarks[23].y],
                            [
                                landmarks[23].x,
                                landmarks[23].y + 1
                            ]
                        )

                    # Right trunk.
                    if is_landmark_visible(
                        landmarks, [12, 24]
                    ):
                        right_trunk_angle = calculate_angle(
                            [landmarks[12].x, landmarks[12].y],
                            [landmarks[24].x, landmarks[24].y],
                            [
                                landmarks[24].x,
                                landmarks[24].y + 1
                            ]
                        )

                    # Pelvis position requires both hip landmarks.
                    if is_landmark_visible(
                        landmarks, [23, 24]
                    ):
                        pelvis_x = (
                            landmarks[23].x + landmarks[24].x
                        ) / 2

                        pelvis_y = (
                            landmarks[23].y + landmarks[24].y
                        ) / 2

                        pelvis_visibility = (
                            landmarks[23].visibility
                            + landmarks[24].visibility
                        ) / 2

                    # Draw landmarks.
                    for landmark in landmarks:

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(
                                frame,
                                (x, y),
                                5,
                                (0, 255, 0),
                                -1
                            )

                    # Draw skeleton.
                    for start, end in connections:

                        start_point = landmarks[start]
                        end_point = landmarks[end]

                        x1 = int(start_point.x * width)
                        y1 = int(start_point.y * height)

                        x2 = int(end_point.x * width)
                        y2 = int(end_point.y * height)

                        if (
                            0 <= x1 < width
                            and 0 <= y1 < height
                            and 0 <= x2 < width
                            and 0 <= y2 < height
                        ):
                            cv2.line(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )

                    # Display valid angles only.
                    display_values = [
                        ("Left Knee", left_knee_angle, 40),
                        ("Right Knee", right_knee_angle, 70),
                        ("Left Hip", left_hip_angle, 100),
                        ("Right Hip", right_hip_angle, 130),
                        ("Left Trunk", left_trunk_angle, 160),
                        ("Right Trunk", right_trunk_angle, 190),
                    ]

                    for label, value, y in display_values:

                        display_text = (
                            f"{label}: {value:.1f} deg"
                            if value is not None
                            else f"{label}: Not visible"
                        )

                        cv2.putText(
                            frame,
                            display_text,
                            (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                # Count valid measurements.
                if left_knee_angle is not None:
                    visibility_counts["left_knee"] += 1

                if right_knee_angle is not None:
                    visibility_counts["right_knee"] += 1

                if (
                    left_knee_angle is not None
                    and right_knee_angle is not None
                ):
                    visibility_counts["both_knees"] += 1

                if left_hip_angle is not None:
                    visibility_counts["left_hip"] += 1

                if right_hip_angle is not None:
                    visibility_counts["right_hip"] += 1

                if (
                    left_hip_angle is not None
                    and right_hip_angle is not None
                ):
                    visibility_counts["both_hips"] += 1

                # Preserve the original CSV columns.
                angle_data.append([
                    frame_number,
                    round(frame_number / fps, 3),

                    left_knee_angle,
                    right_knee_angle,

                    left_hip_angle,
                    right_hip_angle,

                    left_trunk_angle,
                    right_trunk_angle,

                    pelvis_x,
                    pelvis_y,
                    pelvis_visibility
                ])

                out.write(frame)
                frame_number += 1

    finally:
        cap.release()
        out.release()

    try:
        print(
            "Converting processed video to browser-compatible MP4..."
        )

        convert_to_browser_video(
            temporary_video,
            output_video
        )

    except Exception:
        if os.path.exists(temporary_video):
            os.remove(temporary_video)
        raise

    with open(output_csv, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "frame",
            "time_seconds",
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
            "left_trunk_angle",
            "right_trunk_angle",
            "pelvis_x",
            "pelvis_y",
            "pelvis_visibility"
        ])

        writer.writerows(angle_data)

    print("Pose detection completed!")
    print("Frames processed:", frame_number)
    print("Output video:", output_video)
    print("Angle data:", output_csv)

    print("\nLANDMARK VISIBILITY DIAGNOSTICS")
    print("--------------------------------")

    for name, count in visibility_counts.items():
        print(f"{name}: {count} / {frame_number} frames")

    return output_csv


if __name__ == "__main__":

    input_video = "data/samples/squat_test.mp4"
    output_video = "data/samples/squat_pose_output.mp4"
    output_csv = "data/processed/squat_angles.csv"

    analyze_video(
        input_video,
        output_video,
        output_csv
    )
