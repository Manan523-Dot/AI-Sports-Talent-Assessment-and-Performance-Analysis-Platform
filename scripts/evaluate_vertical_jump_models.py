
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "vertical_jump_features.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "ml_evaluation_results.csv"
)

MODEL_DIR = PROJECT_DIR / "data" / "ml"

MODEL_FILE = MODEL_DIR / "vertical_jump_height_model.joblib"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "subject_id",
    "pelvis_baseline_displacement",
    "height_cor_frame_center",
    "well_executed",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Remove identifiers, labels and constant sampling frequency.
excluded_columns = [
    "subject_id",
    "jump_id",
    "sampling_frequency",
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right",
    "well_executed",
]

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
]

X = df[feature_columns]
groups = df["subject_id"]

cv = GroupKFold(n_splits=5)

results = []


# --------------------------------------------------
# REGRESSION: JUMP HEIGHT PREDICTION
# --------------------------------------------------

regression_models = {
    "Dummy baseline": DummyRegressor(strategy="median"),

    "Pelvis linear baseline": LinearRegression(),

    "Ridge regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10)
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ),
}

height_targets = [
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right",
]

print("VERTICAL JUMP ML MODEL EVALUATION")
print("=" * 60)

print("\n1. REGRESSION RESULTS")
print("Using 5-fold participant-grouped cross-validation.")

for target in height_targets:

    valid_df = df[df[target] != -1].copy()

    X_target = valid_df[feature_columns]
    y_target = valid_df[target]
    target_groups = valid_df["subject_id"]

    print(f"\nTARGET: {target}")
    print("-" * 50)

    for model_name, model in regression_models.items():

        predictions = np.zeros(len(valid_df))

        for train_idx, test_idx in cv.split(
            X_target, y_target, target_groups
        ):

            current_model = clone(model)

            if model_name == "Pelvis linear baseline":

                X_train = X_target.iloc[train_idx][
                    ["pelvis_baseline_displacement"]
                ]

                X_test = X_target.iloc[test_idx][
                    ["pelvis_baseline_displacement"]
                ]

            else:
                X_train = X_target.iloc[train_idx]
                X_test = X_target.iloc[test_idx]

            current_model.fit(
                X_train,
                y_target.iloc[train_idx]
            )

            predictions[test_idx] = current_model.predict(X_test)

        mae = mean_absolute_error(y_target, predictions)

        rmse = np.sqrt(
            mean_squared_error(y_target, predictions)
        )

        r2 = r2_score(y_target, predictions)

        print(
            f"{model_name}: "
            f"MAE={mae:.2f} mm, "
            f"RMSE={rmse:.2f} mm, "
            f"R2={r2:.3f}"
        )

        results.append({
            "task": "Regression",
            "target": target,
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })


# --------------------------------------------------
# CLASSIFICATION: MOVEMENT QUALITY
# --------------------------------------------------

print("\n\n2. CLASSIFICATION RESULTS")
print("Target: well_executed")

classification_models = {
    "Dummy baseline": DummyClassifier(strategy="prior"),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
}

y = df["well_executed"].astype(int)

for model_name, model in classification_models.items():

    predictions = np.zeros(len(df), dtype=int)
    probabilities = np.zeros(len(df))

    for train_idx, test_idx in cv.split(X, y, groups):

        current_model = clone(model)

        current_model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        predictions[test_idx] = current_model.predict(
            X.iloc[test_idx]
        )

        probabilities[test_idx] = (
            current_model.predict_proba(X.iloc[test_idx])[:, 1]
        )

    accuracy = accuracy_score(y, predictions)

    balanced_acc = balanced_accuracy_score(y, predictions)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(y, predictions, zero_division=0)

    auc = roc_auc_score(y, probabilities)

    print(f"\n{model_name}")
    print(f"Accuracy:          {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_acc:.3f}")
    print(f"Precision:         {precision:.3f}")
    print(f"Recall:            {recall:.3f}")
    print(f"F1 score:          {f1:.3f}")
    print(f"ROC AUC:           {auc:.3f}")

    results.append({
        "task": "Classification",
        "target": "well_executed",
        "model": model_name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc,
    })


# --------------------------------------------------
# SAVE EVALUATION RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nEvaluation results saved:")
print(OUTPUT_FILE)


# --------------------------------------------------
# TRAIN AND SAVE SELECTED FINAL REGRESSION MODEL
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING SELECTED VERTICAL JUMP MODEL")

# Use only recordings with valid centre-frame height labels.
final_df = df[
    df["height_cor_frame_center"] != -1
].copy()

# Selected feature from candidate validation.
final_feature = ["pelvis_baseline_displacement"]

X_final = final_df[final_feature]

y_final = final_df["height_cor_frame_center"]

# Selected model.
final_model = LinearRegression()

# Train using all valid labelled recordings.
final_model.fit(X_final, y_final)

# Save model and its metadata together.
model_bundle = {
    "model": final_model,
    "feature_names": final_feature,
    "target": "height_cor_frame_center",
    "target_unit": "millimetres",
    "algorithm": "LinearRegression",
    "training_recordings": len(final_df),
    "unique_participants": int(
        final_df["subject_id"].nunique()
    ),
    "dataset": "TU Wien Vertical Jump Dataset",
}

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model_bundle, MODEL_FILE)

print("\nFINAL MODEL TRAINED")
print("-" * 50)
print("Algorithm: Linear Regression")
print("Feature: pelvis_baseline_displacement")
print("Target: height_cor_frame_center")
print(f"Training recordings: {len(final_df)}")
print(
    "Unique participants:",
    final_df["subject_id"].nunique()
)

print("\nModel saved successfully:")
print(MODEL_FILE)

print("\nALL TASKS COMPLETED")
print("=" * 60)
