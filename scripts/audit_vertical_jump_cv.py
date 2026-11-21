
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "vertical_jump_features.csv"
)

df = pd.read_csv(INPUT_FILE)

excluded = [
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
    col for col in df.columns if col not in excluded
]

cv = GroupKFold(n_splits=5)

# --------------------------------------------------
# REGRESSION FOLD AUDIT
# --------------------------------------------------

print("CROSS-VALIDATION AUDIT")
print("=" * 60)

target = "height_cor_frame_center"

valid = df[df[target] != -1].copy()

X = valid[feature_columns]
y = valid[target]
groups = valid["subject_id"]

models = {
    "Pelvis linear baseline": (
        LinearRegression(),
        ["pelvis_baseline_displacement"]
    ),
    "Ridge regression": (
        make_pipeline(
            StandardScaler(),
            Ridge(alpha=10)
        ),
        feature_columns
    ),
}

regression_results = []

print("\n1. HEIGHT PREDICTION: FOLD-BY-FOLD")
print("Target:", target)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    print(f"\nFold {fold}")
    print("Training participants:",
          groups.iloc[train_idx].nunique())
    print("Testing participants:",
          groups.iloc[test_idx].nunique())

    for model_name, (model, columns) in models.items():

        current_model = clone(model)

        current_model.fit(
            X.iloc[train_idx][columns],
            y.iloc[train_idx]
        )

        predictions = current_model.predict(
            X.iloc[test_idx][columns]
        )

        actual = y.iloc[test_idx]

        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)

        print(
            f"{model_name}: "
            f"MAE={mae:.2f} mm, R2={r2:.3f}"
        )

        regression_results.append({
            "fold": fold,
            "model": model_name,
            "MAE": mae,
            "R2": r2,
        })


# --------------------------------------------------
# CLASSIFICATION FOLD AUDIT
# --------------------------------------------------

print("\n\n2. QUALITY CLASSIFICATION: FOLD-BY-FOLD")

X = df[feature_columns]
y = df["well_executed"].astype(int)
groups = df["subject_id"]

classifiers = {
    "Dummy baseline": DummyClassifier(strategy="prior"),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    ),
}

classification_results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):
    print(f"\nFold {fold}")

    print("Test class distribution:")
    print(y.iloc[test_idx].value_counts().to_dict())

    for model_name, model in classifiers.items():

        current_model = clone(model)

        current_model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        predictions = current_model.predict(
            X.iloc[test_idx]
        )

        probabilities = (
            current_model.predict_proba(X.iloc[test_idx])[:, 1]
        )

        actual = y.iloc[test_idx]

        balanced_acc = balanced_accuracy_score(
            actual, predictions
        )

        f1 = f1_score(
            actual, predictions, zero_division=0
        )

        if actual.nunique() == 2:
            auc = roc_auc_score(actual, probabilities)
        else:
            auc = np.nan

        print(
            f"{model_name}: "
            f"Balanced Accuracy={balanced_acc:.3f}, "
            f"F1={f1:.3f}, "
            f"ROC AUC={auc:.3f}"
        )

        classification_results.append({
            "fold": fold,
            "model": model_name,
            "balanced_accuracy": balanced_acc,
            "f1": f1,
            "roc_auc": auc,
        })


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

regression_df = pd.DataFrame(regression_results)
classification_df = pd.DataFrame(classification_results)

print("\n\n3. MEAN FOLD RESULTS")
print("\nRegression:")
print(
    regression_df.groupby("model")[["MAE", "R2"]]
    .agg(["mean", "std"]).round(3)
)

print("\nClassification:")
print(
    classification_df.groupby("model")[
        ["balanced_accuracy", "f1", "roc_auc"]
    ].agg(["mean", "std"]).round(3)
)

regression_df.to_csv(
    PROJECT_DIR / "data" / "processed"
    / "regression_fold_audit.csv",
    index=False
)

classification_df.to_csv(
    PROJECT_DIR / "data" / "processed"
    / "classification_fold_audit.csv",
    index=False
)

print("\nAUDIT COMPLETED")
