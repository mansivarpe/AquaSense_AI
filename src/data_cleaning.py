
from pathlib import Path
import pandas as pd
import numpy as np

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Input CSV file
INPUT_FILE = ROOT / "Data" / "Raw" / "data.csv"

# Output directory
OUTPUT_DIR = ROOT / "Data" / "Processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check that input is a file
if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_FILE}"
    )

# Load dataset
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# Required columns in your current dataset
required = [
    "Timestamp",
    "Tank_Height_cm",
    "Distance_cm",
    "Water_Level_Percentage",
    "Temperature_C",
    "TDS_PPM",
    "Turbidity_NTU",
    "pH_Value",
    "Pump_Status",
    "Water_Quality_Label",
    "Fault_Label",
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# --------------------------------------------------
# TIMESTAMP HANDLING
# --------------------------------------------------

# Current dataset timestamp format: DD-MM-YYYY HH:MM
# Example: 01-10-2026 08:00
#
# Future real-time ESP32 readings:
# Save the actual timestamp for every sensor reading.
# Example: 01-10-2026 08:00:01
#
# Keep this parsing code for future datasets.
# If timestamps include seconds, change the format to:
# "%d-%m-%Y %H:%M:%S"

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# --------------------------------------------------
# REMOVE DUPLICATE TIMESTAMPS (ADDED)
# Keeps the first row for each timestamp
# --------------------------------------------------

rows_before_removal = len(df)

df = df.drop_duplicates(
    subset=["Timestamp"],
    keep="first"
).copy()

df.reset_index(drop=True, inplace=True)

duplicates_removed = rows_before_removal - len(df)

print("\n--- Duplicate Removal ---")
print("Rows before removal:", rows_before_removal)
print("Duplicate rows removed:", duplicates_removed)
print("Rows after removal:", len(df))


# Convert available numeric columns
numeric_columns = [
    "Tank_Height_cm",
    "Distance_cm",
    "Water_Level_Percentage",
    "Temperature_C",
    "TDS_PPM",
    "Turbidity_NTU",
    "pH_Value",
    "Potentiometer_Value",
    "Water_Level_Change_cm_per_min",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create data quality issue column
df["Data_Quality_Issue"] = ""

def flag(mask, message):
    df.loc[mask.fillna(False), "Data_Quality_Issue"] += (
        message + "; "
    )

# Validate readings
flag(df["Timestamp"].isna(), "INVALID_TIMESTAMP")
flag(df["Tank_Height_cm"] <= 0, "INVALID_TANK_HEIGHT")
flag(df["Distance_cm"] < 0, "NEGATIVE_DISTANCE")

flag(
    df["Distance_cm"] > df["Tank_Height_cm"],
    "DISTANCE_EXCEEDS_TANK_HEIGHT"
)

flag(
    ~df["pH_Value"].between(0, 14),
    "PH_OUT_OF_RANGE"
)

flag(df["TDS_PPM"] < 0, "NEGATIVE_TDS")
flag(df["Turbidity_NTU"] < 0, "NEGATIVE_TURBIDITY")

# Calculate water level
# Assumption: Distance_cm represents measured water-level height.
df["Calculated_Water_Level_cm"] = df["Distance_cm"]

# Calculate water level percentage
df["Water_Level_Percentage_Recalculated"] = (
    df["Calculated_Water_Level_cm"]
    / df["Tank_Height_cm"] * 100
).clip(0, 100)

# Flag disagreement with the supplied percentage
df["Percentage_Mismatch"] = (
    (
        df["Water_Level_Percentage"]
        - df["Water_Level_Percentage_Recalculated"]
    ).abs() > 2
)

# Identify timestamps shared by multiple rows
df["Duplicate_Timestamp"] = df["Timestamp"].duplicated(
    keep=False
)

# Save validation report
OUTPUT_FILE = OUTPUT_DIR / "water_validation_report.csv"

df.to_csv(OUTPUT_FILE, index=False)

# Display results
print("\n--- Data Validation Summary ---")
print("Total rows:", len(df))
print("Invalid timestamps:", df["Timestamp"].isna().sum())
print("Invalid tank heights:", (df["Tank_Height_cm"] <= 0).sum())
print("Percentage mismatches:", df["Percentage_Mismatch"].sum())
print("Duplicate timestamps:", df["Duplicate_Timestamp"].sum())
print(
    "Rows with flagged issues:",
    df["Data_Quality_Issue"].ne("").sum()
)

print("\nValidation report saved to:")
print(OUTPUT_FILE)
