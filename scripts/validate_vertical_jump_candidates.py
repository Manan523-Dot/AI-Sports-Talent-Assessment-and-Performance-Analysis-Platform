
from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "vertical_jump_features.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

TARGET = "height_cor_frame_center"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Remove recordings without valid jump-height labels
df = df[df[TARGET] != -1].copy()

df = df.reset_index(drop=True)

groups = df["subject_id"]
y = df[TARGET]


# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {
    "Candidate A - Pelvis only": [
        "pelvis_baseline_displacement",
    ],

    "Candidate B - Normalized pelvis": [
        "pelvis_normalized_vertical_displacement",
    ],

    "Candidate C - Combined pelvis": [
        "pelvis_baseline_displacement",
        "pelvis_normalized_vertical_displacement",
    ],
}


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = GroupKFold(n_splits=5)

all_predictions = []
summary = []

print("VERTICAL JUMP CANDIDATE VALIDATION")
print("=" * 75)

print("Valid recordings:", len(df))
print("Unique participants:", groups.nunique())
print("Target:", TARGET)
print("Cross-validation: 5-fold GroupKFold")
print("Participant leakage: checked")


# ============================================================
# MODEL TRAINING AND EVALUATION
# ============================================================

for model_name, features in feature_sets.items():

    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features for {model_name}: "
            f"{missing_features}"
        )

    X = df[features]

    predictions = pd.Series(
        index=df.index,
        dtype=float
    )

    fold_results = []

    print("\n" + "=" * 75)
    print("MODEL:", model_name)
    print("Features:", features)
    print("-" * 75)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1
    ):

        train_groups = set(
            groups.iloc[train_idx]
        )

        test_groups = set(
            groups.iloc[test_idx]
        )

        overlap = train_groups.intersection(
            test_groups
        )

        if overlap:
            raise ValueError(
                f"Participant leakage detected in fold {fold}"
            )

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearRegression()
        )

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        fold_predictions = model.predict(
            X.iloc[test_idx]
        )

        predictions.iloc[test_idx] = fold_predictions

        actual = y.iloc[test_idx]

        fold_mae = mean_absolute_error(
            actual,
            fold_predictions
        )

        fold_rmse = mean_squared_error(
            actual,
            fold_predictions
        ) ** 0.5

        fold_r2 = r2_score(
            actual,
            fold_predictions
        )

        fold_results.append({
            "Fold": fold,
            "Test recordings": len(test_idx),
            "Test participants": len(test_groups),
            "MAE": fold_mae,
            "RMSE": fold_rmse,
            "R2": fold_r2,
            "Participant overlap": len(overlap),
        })

        print(
            f"Fold {fold}: "
            f"recordings={len(test_idx)}, "
            f"participants={len(test_groups)}, "
            f"MAE={fold_mae:.2f} mm, "
            f"RMSE={fold_rmse:.2f} mm, "
            f"R2={fold_r2:.3f}, "
            f"overlap={len(overlap)}"
        )

    # Ensure every recording received an out-of-fold prediction
    if predictions.isna().any():
        raise ValueError(
            f"Missing predictions for {model_name}"
        )

    # --------------------------------------------------------
    # POOLED OUT-OF-FOLD METRICS
    # --------------------------------------------------------

    pooled_mae = mean_absolute_error(
        y,
        predictions
    )

    pooled_rmse = mean_squared_error(
        y,
        predictions
    ) ** 0.5

    pooled_r2 = r2_score(
        y,
        predictions
    )

    fold_df = pd.DataFrame(fold_results)

    fold_mean_mae = fold_df["MAE"].mean()
    fold_std_mae = fold_df["MAE"].std()

    print("\nFOLD-AVERAGED RESULTS")

    print(
        f"Mean MAE: {fold_mean_mae:.2f} mm"
    )

    print(
        f"MAE standard deviation: {fold_std_mae:.2f} mm"
    )

    print("\nCOMBINED OUT-OF-FOLD RESULTS")

    print(
        f"MAE: {pooled_mae:.2f} mm"
    )

    print(
        f"RMSE: {pooled_rmse:.2f} mm"
    )

    print(
        f"R2: {pooled_r2:.3f}"
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary.append({
        "Model": model_name,
        "Features": ", ".join(features),
        "Fold mean MAE": fold_mean_mae,
        "Fold MAE std": fold_std_mae,
        "Pooled MAE": pooled_mae,
        "Pooled RMSE": pooled_rmse,
        "Pooled R2": pooled_r2,
    })

    # --------------------------------------------------------
    # SAVE OUT-OF-FOLD PREDICTIONS
    # --------------------------------------------------------

    for index in df.index:

        actual_height = float(
            y.loc[index]
        )

        predicted_height = float(
            predictions.loc[index]
        )

        all_predictions.append({
            "model": model_name,
            "subject_id": df.loc[index, "subject_id"],
            "jump_id": df.loc[index, "jump_id"],
            "actual_height_mm": actual_height,
            "predicted_height_mm": predicted_height,
            "absolute_error_mm": abs(
                actual_height - predicted_height
            ),
        })


# ============================================================
# SAVE RESULTS
# ============================================================

summary_df = pd.DataFrame(summary)

predictions_df = pd.DataFrame(
    all_predictions
)

summary_file = (
    OUTPUT_DIR
    / "vertical_jump_candidate_validation.csv"
)

predictions_file = (
    OUTPUT_DIR
    / "vertical_jump_candidate_oof_predictions.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    summary_df[
        [
            "Model",
            "Fold mean MAE",
            "Pooled MAE",
            "Pooled RMSE",
            "Pooled R2",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}"
    )
)

print("\nVALIDATION COMPLETED")

print("Summary:", summary_file)
print("Predictions:", predictions_file)
