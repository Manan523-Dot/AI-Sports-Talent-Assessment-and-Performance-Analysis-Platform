
from pathlib import Path

import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_features.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_regression_predictions.csv"
)

TARGET = "height_cor_frame_center"
BASELINE_FEATURE = "pelvis_baseline_displacement"

df = pd.read_csv(INPUT_FILE)

# Keep only recordings with valid ground-truth height
df = df[df[TARGET] != -1].copy()
df = df.reset_index(drop=True)

# Features must not contain IDs or ground-truth columns
excluded_columns = [
    "subject_id",
    "jump_id",
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right",
    "well_executed"
]

X_all = df.drop(columns=excluded_columns)
X_all = X_all.select_dtypes(include="number")

X_baseline = df[[BASELINE_FEATURE]]
y = df[TARGET]
groups = df["subject_id"]

cv = GroupKFold(n_splits=5)

# Model 1: Linear Regression using pelvis displacement
linear_model = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression()
)

# Model 2: Ridge Regression using all movement features
ridge_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    Ridge(alpha=10)
)

# Generate predictions only for held-out participants
linear_predictions = cross_val_predict(
    linear_model,
    X_baseline,
    y,
    cv=cv,
    groups=groups
)

ridge_predictions = cross_val_predict(
    ridge_model,
    X_all,
    y,
    cv=cv,
    groups=groups
)

def show_metrics(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("-" * 45)
    print("MAE:", round(mae, 2), "mm")
    print("RMSE:", round(rmse, 2), "mm")
    print("R2:", round(r2, 3))


print("VERTICAL JUMP REGRESSION EVALUATION")
print("=" * 55)

print("Valid recordings:", len(df))
print("Unique participants:", groups.nunique())
print("Target:", TARGET)

show_metrics("LINEAR REGRESSION", y, linear_predictions)
show_metrics("RIDGE REGRESSION", y, ridge_predictions)

# Store actual and predicted heights
results = pd.DataFrame({
    "subject_id": df["subject_id"],
    "jump_id": df["jump_id"],
    "actual_height_mm": y,
    "linear_predicted_mm": linear_predictions,
    "ridge_predicted_mm": ridge_predictions
})

results["linear_absolute_error_mm"] = (
    results["actual_height_mm"] -
    results["linear_predicted_mm"]
).abs()

results["ridge_absolute_error_mm"] = (
    results["actual_height_mm"] -
    results["ridge_predicted_mm"]
).abs()

# Show the 10 largest Ridge prediction errors
print("\nTOP 10 LARGEST RIDGE PREDICTION ERRORS")
print("=" * 75)

largest_errors = results.sort_values(
    "ridge_absolute_error_mm",
    ascending=False
).head(10)

print(largest_errors.round(2).to_string(index=False))

# Save all predictions
results.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 55)
print("EVALUATION COMPLETED")
print("Predictions saved to:")
print(OUTPUT_FILE)
