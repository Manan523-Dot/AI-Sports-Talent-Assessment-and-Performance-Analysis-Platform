
from pathlib import Path
import pickle
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR / "data" / "raw" /
    "TUW_2024-JumpMetric_-_Vertical_Jump_Height_Estimation"
)

JOINTS = [
    "pelvis",
    "hip_left", "hip_right",
    "knee_left", "knee_right",
    "ankle_left", "ankle_right"
]


def main():
    files = sorted((DATASET_DIR / "depthcamera").rglob("*.p"))

    print("SKELETON MOVEMENT INSPECTION")
    print("=" * 55)

    for file_path in files[:5]:
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        print(f"\nRecording: {file_path.relative_to(DATASET_DIR)}")
        print("Sampling frequency:", data.get("fs"))

        for joint in JOINTS:
            if joint not in data:
                print(joint, "- MISSING")
                continue

            y = np.asarray(data[joint]["y"], dtype=float)

            print(
                f"{joint:12} "
                f"frames={len(y):3} "
                f"min_y={np.min(y):8.2f} "
                f"max_y={np.max(y):8.2f} "
                f"range={np.ptp(y):8.2f}"
            )

    print("\nINSPECTION COMPLETED")


if __name__ == "__main__":
    main()
