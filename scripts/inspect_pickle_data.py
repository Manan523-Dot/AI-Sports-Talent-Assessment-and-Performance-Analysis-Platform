
from pathlib import Path
import pickle

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_DIR / "data" / "raw" /
    "TUW_2024-JumpMetric_-_Vertical_Jump_Height_Estimation"
)


def inspect_pickle(path, title):
    print(f"\n{'=' * 55}")
    print(title)
    print("File:", path.name)

    with open(path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    print("\nAll top-level keys:")
    print(list(data.keys()))

    print("\nDetailed structure:")

    for key, value in data.items():
        print(f"\n{key}: {type(value).__name__}")

        if isinstance(value, dict):
            print("  Keys:", list(value.keys()))

            for subkey, subvalue in value.items():
                if isinstance(subvalue, (list, tuple)):
                    print(f"  {subkey}: length={len(subvalue)}")

                    if len(subvalue) > 0:
                        print(f"    First 3 values: {subvalue[:3]}")

        elif isinstance(value, (list, tuple)):
            print("  Length:", len(value))
            print("  First 3 values:", value[:3])


def main():
    depth_file = (
        DATASET_DIR / "depthcamera" / "01" / "j1" / "s01_j1.p"
    )

    wearable_file = (
        DATASET_DIR / "wearables" / "01" / "j1" / "s01_j1.p"
    )

    inspect_pickle(depth_file, "DEPTH CAMERA")
    inspect_pickle(wearable_file, "WEARABLE SENSORS")


if __name__ == "__main__":
    main()
