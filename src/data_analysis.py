
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "Data" / "Processed" / "water_validation_report.csv"
REPORT_DIR = ROOT / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Validation report not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 55)
print("WATER MONITORING - EXPLORATORY DATA ANALYSIS")
print("=" * 55)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

# --------------------------------------------------
# 1. DATA TYPES AND MISSING VALUES
# --------------------------------------------------

print("\n--- Data Types ---")
print(df.dtypes)

missing = df.isna().sum().sort_values(ascending=False)

print("\n--- Missing Values ---")
print(missing[missing > 0] if missing.any() else "No missing values")

missing.to_csv(REPORT_DIR / "missing_values.csv")

# --------------------------------------------------
# 2. NUMERIC SUMMARY
# --------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

summary = numeric_df.describe().T
summary["missing_values"] = numeric_df.isna().sum()

print("\n--- Numeric Summary ---")
print(summary.round(3))

summary.to_csv(REPORT_DIR / "numeric_summary.csv")

# --------------------------------------------------
# 3. LABEL DISTRIBUTION
# --------------------------------------------------

for col in ["Water_Quality_Label", "Fault_Label",
            "Pump_Status", "Buzzer_Status", "LED_Status"]:
    if col in df.columns:
        print(f"\n--- {col} Distribution ---")
        print(df[col].value_counts(dropna=False))

        counts = df[col].fillna("MISSING").astype(str).value_counts()

        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Number of records")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"{col.lower()}_distribution.png", dpi=150)
        plt.close()

# --------------------------------------------------
# 4. SENSOR HISTOGRAMS
# --------------------------------------------------

sensor_columns = [
    "Water_Level_Percentage",
    "Temperature_C",
    "TDS_PPM",
    "Turbidity_NTU",
    "pH_Value",
    "Potentiometer_Value",
]

for col in sensor_columns:
    if col not in df.columns:
        continue

    values = pd.to_numeric(df[col], errors="coerce").dropna()

    if values.empty:
        print(f"Skipping empty column: {col}")
        continue

    plt.figure(figsize=(7, 4))
    sns.histplot(values, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{col.lower()}_histogram.png", dpi=150)
    plt.close()

# --------------------------------------------------
# 5. SENSOR TRENDS OVER TIME
# --------------------------------------------------

if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce", dayfirst=True
    )
    df = df.sort_values("Timestamp")

    print("\n--- Timestamp Analysis ---")
    print("First timestamp:", df["Timestamp"].min())
    print("Last timestamp:", df["Timestamp"].max())
    print("Duplicate timestamps:", df["Timestamp"].duplicated().sum())

    time_columns = [
        "Water_Level_Percentage",
        "Temperature_C",
        "TDS_PPM",
        "Turbidity_NTU",
        "pH_Value",
    ]

    for col in time_columns:
        if col not in df.columns:
            continue

        plot_df = df[["Timestamp", col]].copy()
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna()

        if plot_df.empty:
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(plot_df["Timestamp"], plot_df[col], marker=".", linewidth=1)
        plt.title(f"{col} Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel(col)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"{col.lower()}_over_time.png", dpi=150)
        plt.close()

# --------------------------------------------------
# 6. CORRELATION ANALYSIS
# --------------------------------------------------

corr_columns = [
    "Water_Level_Percentage",
    "Temperature_C",
    "TDS_PPM",
    "Turbidity_NTU",
    "pH_Value",
    "Potentiometer_Value",
]

available = [c for c in corr_columns if c in df.columns]

if len(available) >= 2:
    corr = df[available].apply(pd.to_numeric, errors="coerce").corr()

    print("\n--- Correlation Matrix ---")
    print(corr.round(3))

    corr.to_csv(REPORT_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(9, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Sensor Feature Correlation")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

# --------------------------------------------------
# 7. PHYSICAL AND DATA QUALITY CHECKS
# --------------------------------------------------

print("\n--- Range Checks ---")

checks = {
    "Tank height <= 0": (
        "Tank_Height_cm", lambda s: s <= 0
    ),
    "Negative distance": (
        "Distance_cm", lambda s: s < 0
    ),
    "pH outside 0-14": (
        "pH_Value", lambda s: ~s.between(0, 14)
    ),
    "Negative TDS": (
        "TDS_PPM", lambda s: s < 0
    ),
    "Negative turbidity": (
        "Turbidity_NTU", lambda s: s < 0
    ),
}

for name, (col, condition) in checks.items():
    if col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        print(f"{name}: {condition(values).fillna(False).sum()}")

# --------------------------------------------------
# 8. SAVE FINAL EDA SUMMARY
# --------------------------------------------------

with open(REPORT_DIR / "eda_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Dataset rows: {len(df)}\n")
    f.write(f"Dataset columns: {len(df.columns)}\n")
    f.write(f"Missing cells: {int(df.isna().sum().sum())}\n")

    if "Timestamp" in df.columns:
        f.write(f"First timestamp: {df['Timestamp'].min()}\n")
        f.write(f"Last timestamp: {df['Timestamp'].max()}\n")
        f.write(
            f"Duplicate timestamps: {int(df['Timestamp'].duplicated().sum())}\n"
        )

print("\nEDA completed successfully!")
print("Graphs and reports saved to:", REPORT_DIR)
