
from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "vertical_jump_features.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "stratified_classification_results.csv"
)

df = pd.read_csv(INPUT_FILE)

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
y = df["well_executed"].astype(int)
groups = df["subject_id"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
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

results = []

print("STRATIFIED GROUP CLASSIFICATION")
print("=" * 60)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    print(f"\nFOLD {fold}")
    print("-" * 45)

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    print("Training participants:", len(train_groups))
    print("Testing participants:", len(test_groups))
    print("Participant overlap:", len(train_groups & test_groups))

    actual = y.iloc[test_idx]

    print(
        "Test class distribution:",
        actual.value_counts().to_dict()
    )

    for model_name, model in models.items():

        current_model = clone(model)

        current_model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        predictions = current_model.predict(
            X.iloc[test_idx]
        )

        probabilities = (
            current_model.predict_proba(
                X.iloc[test_idx]
            )[:, 1]
        )

        balanced_acc = balanced_accuracy_score(
            actual, predictions
        )

        accuracy = accuracy_score(
            actual, predictions
        )

        precision = precision_score(
            actual, predictions,
            zero_division=0
        )

        recall = recall_score(
            actual, predictions,
            zero_division=0
        )

        f1 = f1_score(
            actual, predictions,
            zero_division=0
        )

        auc = roc_auc_score(actual, probabilities)

        print(f"\n{model_name}")
        print(f"Accuracy:          {accuracy:.3f}")
        print(f"Balanced accuracy: {balanced_acc:.3f}")
        print(f"Precision:         {precision:.3f}")
        print(f"Recall:            {recall:.3f}")
        print(f"F1 score:          {f1:.3f}")
        print(f"ROC AUC:           {auc:.3f}")

        results.append({
            "fold": fold,
            "model": model_name,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
        })

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("MEAN CROSS-VALIDATION RESULTS")

print(
    results_df.groupby("model")[
        ["accuracy", "balanced_accuracy", "precision",
         "recall", "f1", "roc_auc"]
    ].mean().round(3)
)

print("\nResults saved to:")
print(OUTPUT_FILE)
