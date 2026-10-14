
from pathlib import Path

import joblib
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "models" / "anomaly_model.joblib"
INPUT_FILE = ROOT / "Data" / "Raw" / "data.csv"
OUTPUT_DIR = ROOT / "Data" / "Processed"
OUTPUT_FILE = OUTPUT_DIR / "prediction_report.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These must match the features used during training.
FEATURES = [
    "Water_Level_Percentage",
    "Temperature_C",
    "TDS_PPM",
    "Turbidity_NTU",
    "pH_Value",
]

# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------

if not MODEL_FILE.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

if not INPUT_FILE.is_file():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

model = joblib.load(MODEL_FILE)
df = pd.read_csv(INPUT_FILE)

df.columns = df.columns.str.strip()

missing = [col for col in FEATURES if col not in df.columns]

if missing:
    raise ValueError(f"Missing model features: {missing}")

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

# Do not make predictions for incomplete records.
valid_mask = X.notna().all(axis=1)
X_valid = X.loc[valid_mask]

if X_valid.empty:
    raise ValueError("No complete rows are available for prediction.")

# --------------------------------------------------
# PREDICT ANOMALIES
# --------------------------------------------------

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

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
# GENERATE REVIEW ALERTS
# --------------------------------------------------

df["Review_Alert"] = "No anomaly flagged"

df.loc[
    df["Anomaly_Label"] == "Anomaly",
    "Review_Alert"
] = "Review sensor readings"

df.loc[
    ~valid_mask,
    "Review_Alert"
] = "Invalid or missing model features"

# Save every record, including records that could not be scored.
df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# DISPLAY SUMMARY
# --------------------------------------------------

print("\nPREDICTION COMPLETED")
print("-" * 40)
print("Total records:", len(df))
print("Records scored:", int(valid_mask.sum()))
print("Records not scored:", int((~valid_mask).sum()))

print("\nPrediction summary:")
print(df["Anomaly_Label"].value_counts(dropna=False))

print("\nReview alert summary:")
print(df["Review_Alert"].value_counts(dropna=False))

print("\nFirst 10 results:")
columns_to_show = FEATURES + [
    "Anomaly_Score",
    "Anomaly_Label",
    "Review_Alert"
]

print(df[columns_to_show].head(10).to_string(index=False))

print("\nSaved prediction report:")
print(OUTPUT_FILE)
