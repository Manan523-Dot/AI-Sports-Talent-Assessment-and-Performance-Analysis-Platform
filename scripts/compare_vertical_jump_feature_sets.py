
from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "vertical_jump_features.csv"
)

TARGET = "height_cor_frame_center"

df = pd.read_csv(INPUT_FILE)

# Only recordings with valid jump-height labels
df = df[df[TARGET] != -1].copy()

excluded_columns = [
    "subject_id",
    "jump_id",
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right",
    "well_executed",
]

X = df.drop(columns=excluded_columns)
X = X.select_dtypes(include="number")

# Remove constant features
X = X.loc[:, X.nunique() > 1]

y = df[TARGET]
groups = df["subject_id"]

# Define feature combinations before model evaluation
feature_sets = {
    "Pelvis only": [
        "pelvis_baseline_displacement",
    ],

    "Pelvis + chest": [
        "pelvis_baseline_displacement",
        "chest_baseline_displacement",
    ],

    "Pelvis movement": [
        "pelvis_baseline_displacement",
        "pelvis_vertical_std",
        "pelvis_vertical_range",
    ],

    "Pelvis + chest movement": [
        "pelvis_baseline_displacement",
        "chest_baseline_displacement",
        "pelvis_vertical_std",
        "pelvis_vertical_range",
        "chest_vertical_std",
        "chest_vertical_range",
    ],

    "Pelvis + knee": [
        "pelvis_baseline_displacement",
        "left_knee_angle_mean",
        "right_knee_angle_mean",
        "left_knee_angle_range",
        "right_knee_angle_range",
    ],

    "All features": list(X.columns),
}

# Compare Linear Regression and Ridge on all features
experiments = [
    ("Pelvis only", LinearRegression()),
    ("Pelvis + chest", LinearRegression()),
    ("Pelvis movement", LinearRegression()),
    ("Pelvis + chest movement", LinearRegression()),
    ("Pelvis + knee", LinearRegression()),
    ("All features - Linear", LinearRegression()),
    ("All features - Ridge", Ridge(alpha=10)),
]

cv = GroupKFold(n_splits=5)

results = []

print("VERTICAL JUMP FEATURE SET COMPARISON")
print("=" * 75)

print("\nDATASET")
print("Valid recordings:", len(df))
print("Unique participants:", groups.nunique())
print("Target:", TARGET)

# Check redundancy among the strongest features
top_features = [
    "pelvis_baseline_displacement",
    "chest_baseline_displacement",
    "pelvis_vertical_std",
    "pelvis_vertical_range",
    "chest_vertical_std",
    "chest_vertical_range",
]

top_features = [f for f in top_features if f in X.columns]

print("\n1. CORRELATION BETWEEN STRONG FEATURES")
correlation_matrix = X[top_features].corr()

for i in range(len(top_features)):
    for j in range(i + 1, len(top_features)):
        feature_a = top_features[i]
        feature_b = top_features[j]
        correlation = correlation_matrix.loc[feature_a, feature_b]

        if abs(correlation) >= 0.80:
            print(
                f"{feature_a} <-> {feature_b}: "
                f"{correlation:.3f}"
            )

# Evaluate every experiment using identical participant-grouped folds
print("\n2. FIVE-FOLD GROUPED MODEL COMPARISON")
print("-" * 75)

for experiment_name, model in experiments:

    if experiment_name.startswith("All features"):
        selected_features = feature_sets["All features"]
    else:
        selected_features = feature_sets[experiment_name]

    selected_features = [
        feature for feature in selected_features
        if feature in X.columns
    ]

    model_X = X[selected_features]

    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for train_index, test_index in cv.split(
        model_X, y, groups
    ):
        X_train = model_X.iloc[train_index]
        X_test = model_X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            model
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        fold_mae.append(
            mean_absolute_error(y_test, predictions)
        )

        fold_rmse.append(
            mean_squared_error(y_test, predictions) ** 0.5
        )

        fold_r2.append(
            r2_score(y_test, predictions)
        )

    results.append({
        "Model": experiment_name,
        "Features": len(selected_features),
        "Mean MAE": sum(fold_mae) / 5,
        "MAE Std": pd.Series(fold_mae).std(ddof=0),
        "Mean RMSE": sum(fold_rmse) / 5,
        "Mean R2": sum(fold_r2) / 5,
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Mean MAE")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Mean MAE": "{:.2f}".format,
            "MAE Std": "{:.2f}".format,
            "Mean RMSE": "{:.2f}".format,
            "Mean R2": "{:.3f}".format,
        }
    )
)

# Save results
output_file = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "vertical_jump_feature_set_comparison.csv"
)

results_df.to_csv(output_file, index=False)

print("\n" + "=" * 75)
print("ANALYSIS COMPLETED")
print("Saved:", output_file)
