
from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_features.csv"
)

df = pd.read_csv(INPUT_FILE)

TARGET = "height_cor_frame_center"

# Keep only valid target records
df = df[df[TARGET] != -1].copy()

excluded = [
    "subject_id",
    "jump_id",
    "height_measured",
    "height_cor_frame_left",
    "height_cor_frame_center",
    "height_cor_frame_right",
    "well_executed"
]

X = df.drop(columns=excluded)
X = X.select_dtypes(include="number")
y = df[TARGET]
groups = df["subject_id"]

# Remove constant features
X = X.loc[:, X.nunique() > 1]

print("REGRESSION FEATURE ANALYSIS")
print("=" * 65)

# Correlation with target
correlations = X.corrwith(y).sort_values(
    key=abs,
    ascending=False
)

print("\n1. FEATURE CORRELATION WITH JUMP HEIGHT")
print(correlations.round(3).to_string())

# Ridge coefficient analysis
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    Ridge(alpha=10)
)

cv = GroupKFold(n_splits=5)

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    groups=groups,
    scoring="neg_mean_absolute_error"
)

pipeline.fit(X, y)

coefficients = pd.Series(
    pipeline.named_steps["ridge"].coef_,
    index=X.columns
)

coefficients = coefficients.reindex(
    coefficients.abs().sort_values(ascending=False).index
)

print("\n2. RIDGE FEATURE COEFFICIENTS")
print(coefficients.round(3).to_string())

print("\n3. FIVE-FOLD GROUPED VALIDATION")
print("Mean MAE:", round(-scores.mean(), 2), "mm")
print("Fold MAEs:", (-scores).round(2))

output_file = (
    PROJECT_DIR / "data" / "processed" /
    "vertical_jump_feature_analysis.csv"
)

pd.DataFrame({
    "correlation": correlations,
    "ridge_coefficient": coefficients
}).to_csv(output_file)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("Saved:", output_file)
