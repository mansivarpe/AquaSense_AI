
from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "Data" / "Processed" / "water_validation_report.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = ROOT / "Data" / "Processed"
OUTPUT_FILE = OUTPUT_DIR / "anomaly_predictions.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if not INPUT_FILE.is_file():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

FEATURES = [
    "Water_Level_Percentage",
    "Temperature_C",
    "TDS_PPM",
    "Turbidity_NTU",
    "pH_Value",
]

missing = [col for col in FEATURES if col not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

# Exclude invalid rows from training and prediction.
valid_mask = X.notna().all(axis=1)

if valid_mask.sum() < 100:
    raise ValueError(
        "At least 100 complete readings are required "
        "for this initial training run."
    )

X_valid = X.loc[valid_mask]

# --------------------------------------------------
# BUILD AND TRAIN MODEL
# --------------------------------------------------

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42
    )
)

model.fit(X_valid)

# --------------------------------------------------
# GENERATE ANOMALY FLAGS
# --------------------------------------------------

df["Anomaly_Result"] = pd.Series(
    pd.NA, index=df.index, dtype="Int64"
)
df["Anomaly_Score"] = float("nan")

df.loc[valid_mask, "Anomaly_Result"] = model.predict(X_valid)
df.loc[valid_mask, "Anomaly_Score"] = (
    model.decision_function(X_valid)
)

df["Anomaly_Label"] = df["Anomaly_Result"].map({
    1: "Normal",
    -1: "Anomaly"
})

# --------------------------------------------------
# SAVE MODEL AND RESULTS
# --------------------------------------------------

model_path = MODEL_DIR / "anomaly_model.joblib"

joblib.dump(model, model_path)
df.to_csv(OUTPUT_FILE, index=False)

print("\nMODEL TRAINING COMPLETED")
print("------------------------")
print("Total dataset rows:", len(df))
print("Rows used for training:", len(X_valid))
print("Model saved:", model_path)
print("Predictions saved:", OUTPUT_FILE)

print("\nAnomaly results:")
print(df["Anomaly_Label"].value_counts(dropna=False))

print("\nFirst 10 predictions:")
print(
    df[
        FEATURES + ["Anomaly_Score", "Anomaly_Label"]
    ].head(10).to_string(index=False)
)
