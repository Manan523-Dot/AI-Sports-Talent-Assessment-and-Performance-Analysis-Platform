
from pathlib import Path
import pickle
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR / "data" / "raw" /
    "TUW_2024-JumpMetric_-_Vertical_Jump_Height_Estimation"
)

JOINTS = ["pelvis", "spine_chest"]


def show_joint(data, joint):
    y = np.asarray(data[joint]["y"], dtype=float)

    min_index = int(np.argmin(y))
    max_index = int(np.argmax(y))

    print(f"\n{joint.upper()}")
    print("-" * 40)
    print("Total frames:", len(y))
    print("Highest position (minimum Y):", round(y[min_index], 2))
    print("Frame of highest position:", min_index)
    print("Lowest position (maximum Y):", round(y[max_index], 2))
    print("Frame of lowest position:", max_index)

    print("\nFirst 10 Y values:")
    print(np.round(y[:10], 2).tolist())

    print("\nLast 10 Y values:")
    print(np.round(y[-10:], 2).tolist())

    print("\nValues around highest position:")
    print(np.round(y[max(0, min_index - 3):min(len(y), min_index + 4)], 2).tolist())


def main():
    files = sorted((DATASET_DIR / "depthcamera").rglob("*.p"))

    print("JUMP TRAJECTORY INSPECTION")
    print("=" * 50)

    for file_path in files[:5]:
        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        print("\n" + "=" * 50)
        print("Recording:", file_path.relative_to(DATASET_DIR))
        print("Sampling frequency:", data.get("fs"))

        for joint in JOINTS:
            show_joint(data, joint)

    print("\nTRAJECTORY INSPECTION COMPLETED")


if __name__ == "__main__":
    main()
