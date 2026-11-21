
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# Project paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "datasets",
    "vertical_jump_training.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "vertical_jump"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_model.joblib"
)


# Features used by the ML model
FEATURE_COLUMNS = [
    "estimated_jump_height_cm",
    "flight_time",
    "knee_range_of_motion",
    "knee_rom_difference",
    "takeoff_knee_angle",
    "landing_knee_angle",
    "max_knee_angle",
    "min_knee_angle",
    "left_knee_rom",
    "right_knee_rom",
    "jump_detected"
]

TARGET_COLUMN = "performance_score"


def train_model():

    print("\n===== VERTICAL JUMP ML TRAINING =====")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Training dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    # Convert values to numeric
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    # Validate dataset size
    if len(df) < 20:
        raise ValueError(
            "At least 20 valid labelled records are required "
            "for this initial training experiment. "
            "More varied records are strongly recommended."
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    if not y.between(0, 100).all():
        raise ValueError(
            "Performance scores must be between 0 and 100."
        )

    # Training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )

    print("Training Random Forest model...")

    model.fit(X_train, y_train)

    # Evaluate on unseen test data
    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Save model and training information
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_data = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "mae": float(mae),
        "r2_score": float(r2),
        "model_version": "1.0"
    }

    joblib.dump(
        model_data,
        MODEL_PATH
    )

    print("\nTraining completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.3f}")
    print(f"Model saved: {MODEL_PATH}")

    print("====================================\n")


if __name__ == "__main__":
    train_model()
