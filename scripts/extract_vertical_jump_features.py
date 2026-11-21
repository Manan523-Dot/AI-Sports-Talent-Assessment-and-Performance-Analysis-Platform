
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "TUW_2024-JumpMetric_-_Vertical_Jump_Height_Estimation"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "vertical_jump_features.csv"

JOINTS = [
    "pelvis",
    "spine_chest",
    "spine_navel",
    "hip_left",
    "hip_right",
    "knee_left",
    "knee_right",
    "ankle_left",
    "ankle_right",
]


def get_joint(data, joint):
    """Return a joint's 3D coordinates as an array of shape (frames, 3)."""
    return np.column_stack([
        np.asarray(data[joint][axis], dtype=float)
        for axis in ["x", "y", "z"]
    ])


def calculate_angle(a, b, c):
    """Calculate the angle ABC in degrees for every frame."""
    ba = a - b
    bc = c - b

    dot = np.sum(ba * bc, axis=1)

    denominator = (
        np.linalg.norm(ba, axis=1)
        * np.linalg.norm(bc, axis=1)
    )

    cosine = np.divide(
        dot,
        denominator,
        out=np.full_like(dot, np.nan, dtype=float),
        where=denominator > 1e-8
    )

    return np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )


def add_vertical_features(features, coordinates, name):
    """Extract vertical movement features."""
    y = coordinates[:, 1]

    baseline_start = np.median(y[:15])
    baseline_end = np.median(y[-15:])
    baseline = (baseline_start + baseline_end) / 2

    peak_index = int(np.argmin(y))

    features[f"{name}_vertical_range"] = float(np.ptp(y))
    features[f"{name}_vertical_std"] = float(np.std(y))

    features[f"{name}_start_end_baseline_difference"] = float(
        abs(baseline_start - baseline_end)
    )

    features[f"{name}_baseline_displacement"] = float(
        baseline - y[peak_index]
    )

    features[f"{name}_peak_time_ratio"] = float(
        peak_index / max(len(y) - 1, 1)
    )


def add_normalized_pelvis_feature(features, coordinates):
    """
    Calculate pelvis vertical displacement relative to body height.

    The displacement is divided by the median vertical distance
    between the chest and the midpoint of the ankles.

    This produces a dimensionless feature rather than a measurement
    in millimetres. It is a candidate feature for later comparison
    with normalized MediaPipe image coordinates.
    """

    pelvis_y = coordinates["pelvis"][:, 1]
    chest_y = coordinates["spine_chest"][:, 1]

    ankle_midpoint_y = (
        coordinates["ankle_left"][:, 1]
        + coordinates["ankle_right"][:, 1]
    ) / 2

    baseline_start = np.median(pelvis_y[:15])
    baseline_end = np.median(pelvis_y[-15:])
    baseline = (baseline_start + baseline_end) / 2

    peak_y = np.min(pelvis_y)

    body_vertical_scale = np.median(
        np.abs(chest_y - ankle_midpoint_y)
    )

    if not np.isfinite(body_vertical_scale) or body_vertical_scale <= 1e-8:
        raise ValueError("Invalid body vertical scale")

    normalized_displacement = (
        (baseline - peak_y) / body_vertical_scale
    )

    features["pelvis_normalized_vertical_displacement"] = float(
        normalized_displacement
    )


def add_angle_features(features, angles, name):
    """Extract statistics from a joint-angle sequence."""
    valid = angles[np.isfinite(angles)]

    if len(valid) == 0:
        raise ValueError(f"No valid angle values for {name}")

    features[f"{name}_minimum"] = float(np.min(valid))
    features[f"{name}_maximum"] = float(np.max(valid))
    features[f"{name}_mean"] = float(np.mean(valid))
    features[f"{name}_std"] = float(np.std(valid))
    features[f"{name}_range"] = float(np.ptp(valid))


def process_recording(file_path, groundtruth):
    subject_id = int(file_path.parent.parent.name)
    jump_id = int(file_path.parent.name.replace("j", ""))

    matching = groundtruth[
        (groundtruth["id"] == subject_id)
        & (groundtruth["jump"] == jump_id)
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Could not uniquely match labels: "
            f"subject {subject_id}, jump {jump_id}"
        )

    label = matching.iloc[0]

    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    missing = [
        joint for joint in JOINTS
        if joint not in data
    ]

    if missing:
        raise ValueError(
            f"Missing joints in {file_path}: {missing}"
        )

    coordinates = {
        joint: get_joint(data, joint)
        for joint in JOINTS
    }

    frame_counts = {
        len(values)
        for values in coordinates.values()
    }

    if len(frame_counts) != 1:
        raise ValueError(
            f"Joint frame counts do not match: {file_path}"
        )

    if not all(
        np.all(np.isfinite(values))
        for values in coordinates.values()
    ):
        raise ValueError(
            f"Non-finite coordinates: {file_path}"
        )

    frame_count = len(coordinates["pelvis"])
    fs = float(data["fs"])

    features = {
        "subject_id": subject_id,
        "jump_id": jump_id,
        "frame_count": frame_count,
        "sampling_frequency": fs,
        "duration_seconds": frame_count / fs,
    }

    # Pelvis and chest movement
    add_vertical_features(
        features,
        coordinates["pelvis"],
        "pelvis"
    )

    add_vertical_features(
        features,
        coordinates["spine_chest"],
        "chest"
    )

    # New body-normalized pelvis movement feature
    add_normalized_pelvis_feature(
        features,
        coordinates
    )

    # Horizontal and depth movement
    for joint in ["pelvis", "spine_chest"]:
        points = coordinates[joint]

        features[f"{joint}_x_range"] = float(
            np.ptp(points[:, 0])
        )

        features[f"{joint}_z_range"] = float(
            np.ptp(points[:, 2])
        )

    # Knee angles
    left_knee = calculate_angle(
        coordinates["hip_left"],
        coordinates["knee_left"],
        coordinates["ankle_left"]
    )

    right_knee = calculate_angle(
        coordinates["hip_right"],
        coordinates["knee_right"],
        coordinates["ankle_right"]
    )

    add_angle_features(
        features,
        left_knee,
        "left_knee_angle"
    )

    add_angle_features(
        features,
        right_knee,
        "right_knee_angle"
    )

    # Left-right knee movement comparison
    features["knee_angle_range_difference"] = float(
        abs(np.ptp(left_knee) - np.ptp(right_knee))
    )

    features["knee_mean_angle_difference"] = float(
        abs(np.nanmean(left_knee) - np.nanmean(right_knee))
    )

    # Ground-truth labels (targets, not input features)
    features["height_measured"] = float(
        label["height_measured"]
    )

    features["height_cor_frame_left"] = float(
        label["height_cor_frame_left"]
    )

    features["height_cor_frame_center"] = float(
        label["height_cor_frame_center"]
    )

    features["height_cor_frame_right"] = float(
        label["height_cor_frame_right"]
    )

    features["well_executed"] = int(
        label["well_executed"]
    )

    return features


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    groundtruth = pd.read_csv(
        DATASET_DIR / "groundtruth.csv",
        sep=";"
    )

    files = sorted(
        (DATASET_DIR / "depthcamera").rglob("*.p")
    )

    records = []

    print("VERTICAL JUMP FEATURE EXTRACTION")
    print("=" * 55)
    print("Recordings found:", len(files))

    for index, file_path in enumerate(files, start=1):
        record = process_recording(
            file_path,
            groundtruth
        )

        records.append(record)

        if index % 25 == 0 or index == len(files):
            print(
                f"Processed {index}/{len(files)} recordings"
            )

    df = pd.DataFrame(records)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFEATURE EXTRACTION COMPLETED")
    print("Total recordings:", len(df))
    print("Total columns:", len(df.columns))
    print("Output:", OUTPUT_FILE)

    print("\nNew normalized feature:")
    print(
        df["pelvis_normalized_vertical_displacement"]
        .describe()
    )

    print("\nFeature table preview:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nHeight records with valid labels:")
    print(
        (df["height_cor_frame_center"] != -1).sum()
    )

    print("\nQuality label distribution:")
    print(
        df["well_executed"].value_counts().sort_index()
    )


if __name__ == "__main__":
    main()
