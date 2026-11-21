
from pathlib import Path
import pickle
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR / "data" / "raw" /
    "TUW_2024-JumpMetric_-_Vertical_Jump_Height_Estimation"
)

JOINTS = [
    "pelvis", "spine_navel", "spine_chest", "neck",
    "hip_left", "hip_right",
    "knee_left", "knee_right",
    "ankle_left", "ankle_right",
    "foot_left", "foot_right"
]


def main():
    files = sorted((DATASET_DIR / "depthcamera").rglob("*.p"))

    print("DATASET QUALITY CHECK")
    print("=" * 45)
    print("Depth-camera recordings:", len(files))

    fs_values = []
    observation_counts = []
    invalid_count = 0

    for file_path in files:
        with open(file_path, "rb") as file:
            data = pickle.load(file, encoding="latin1")

        fs = data.get("fs")
        if fs is not None:
            fs_values.append(float(fs))

        for joint in JOINTS:
            if joint not in data:
                invalid_count += 1
                continue

            lengths = []

            for axis in ["x", "y", "z"]:
                values = np.asarray(data[joint][axis], dtype=float)
                lengths.append(len(values))

                if not np.all(np.isfinite(values)):
                    invalid_count += 1

            if len(set(lengths)) != 1:
                invalid_count += 1

            observation_counts.append(lengths[0])

    print("\nSampling frequency values:")
    print(sorted(set(fs_values)))

    print("\nObservation counts:")
    print("Minimum:", min(observation_counts))
    print("Maximum:", max(observation_counts))

    print("\nMissing/invalid checks:")
    print("Issues detected:", invalid_count)

    print("\nQUALITY CHECK COMPLETED")


if __name__ == "__main__":
    main()
