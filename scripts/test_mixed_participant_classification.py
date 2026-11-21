
from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "vertical_jump_features.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only participants who have both label types
label_counts = df.groupby("subject_id")["well_executed"].nunique()
mixed_ids = label_counts[label_counts > 1].index

mixed_df = df[df["subject_id"].isin(mixed_ids)].copy()

# Exclude IDs and ground-truth columns from model inputs
excluded = [
    "subject_id",
    "jump_id",
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right",
    "well_executed"
]

X = mixed_df.drop(columns=excluded)
X = X.select_dtypes(include="number")

y = mixed_df["well_executed"].astype(int)
groups = mixed_df["subject_id"]

print("MIXED-PARTICIPANT CLASSIFICATION TEST")
print("=" * 55)
print("Participants:", len(mixed_ids))
print("Recordings:", len(mixed_df))
print("Features:", X.shape[1])
print("\nLabel distribution:")
print(y.value_counts().sort_index())

cv = LeaveOneGroupOut()

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced")
)

dummy = DummyClassifier(strategy="most_frequent")

# Each participant is held out as a complete test group
predictions = cross_val_predict(
    model, X, y, groups=groups, cv=cv
)

probabilities = cross_val_predict(
    model, X, y, groups=groups, cv=cv, method="predict_proba"
)[:, 1]

dummy_predictions = cross_val_predict(
    dummy, X, y, groups=groups, cv=cv
)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", round(accuracy_score(y, predictions), 3))
print("Balanced accuracy:", round(balanced_accuracy_score(y, predictions), 3))
print("F1-score:", round(f1_score(y, predictions), 3))
print("ROC AUC:", round(roc_auc_score(y, probabilities), 3))
print("Confusion matrix:")
print(confusion_matrix(y, predictions))

print("\nDUMMY BASELINE")
print("Accuracy:", round(accuracy_score(y, dummy_predictions), 3))
print("Balanced accuracy:", round(
    balanced_accuracy_score(y, dummy_predictions), 3
))

print("\n" + "=" * 55)
print("EXPERIMENT COMPLETED")
