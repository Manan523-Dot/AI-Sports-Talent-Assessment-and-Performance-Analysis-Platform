import csv
import json
import os


FEATURES_FOLDER = "data/ml/squat_features"
OUTPUT_FILE = "data/ml/squat_dataset.csv"


FIELDNAMES = [
    "number_of_reps",
    "average_knee_rom",
    "average_bottom_angle",
    "average_rep_time",
    "average_descent_time",
    "average_ascent_time",
    "consistency",
    "label",
]


def extract_features(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    return {
        "number_of_reps": data["repetitions"],
        "average_knee_rom": data["knee_rom"],
        "average_bottom_angle": data["average_rep_bottom_angle"],
        "average_rep_time": data["average_rep_time"],
        "average_descent_time": data["average_descent_time"],
        "average_ascent_time": data["average_ascent_time"],
        "consistency": data["rep_time_difference"],
        "label": "unknown",
    }


def build_squat_dataset():
    if not os.path.exists(FEATURES_FOLDER):
        raise FileNotFoundError(
            f"Feature folder not found: {FEATURES_FOLDER}"
        )

    feature_files = sorted(
        file
        for file in os.listdir(FEATURES_FOLDER)
        if file.lower().endswith(".json")
    )

    if not feature_files:
        raise ValueError(
            "No Squat feature JSON files found."
        )

    rows = []

    for filename in feature_files:
        file_path = os.path.join(
            FEATURES_FOLDER,
            filename
        )

        try:
            row = extract_features(file_path)
            rows.append(row)
            print(f"Processed: {filename}")

        except Exception as error:
            print(
                f"Skipped {filename}: {error}"
            )

    if not rows:
        raise ValueError(
            "No valid Squat feature files could be processed."
        )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSquat dataset rebuilt successfully."
    )
    print(
        f"Total samples: {len(rows)}"
    )
    print(
        f"Dataset saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    build_squat_dataset()