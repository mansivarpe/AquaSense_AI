
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "Data" / "Processed" / "water_validation_report.csv"

df = pd.read_csv(FILE)

print("\n--- FIRST 10 ROWS ---")
print(df[
    [
        "Timestamp",
        "Tank_Height_cm",
        "Distance_cm",
        "Water_Level_Percentage",
        "Water_Level_Percentage_Recalculated",
        "Duplicate_Timestamp",
    ]
].head(10).to_string(index=False))

print("\n--- TIMESTAMP SUMMARY ---")
print("Total rows:", len(df))
print("Unique timestamps:", df["Timestamp"].nunique())
print("Rows with duplicate timestamps:", df["Timestamp"].duplicated(keep=False).sum())

print("\n--- PERCENTAGE COMPARISON ---")
print(
    df[
        [
            "Water_Level_Percentage",
            "Water_Level_Percentage_Recalculated",
            "Percentage_Mismatch",
        ]
    ].describe()
)

print("\n--- SAMPLE MISMATCHES ---")
print(
    df.loc[
        df["Percentage_Mismatch"],
        [
            "Tank_Height_cm",
            "Distance_cm",
            "Water_Level_Percentage",
            "Water_Level_Percentage_Recalculated",
        ],
    ].head(10).to_string(index=False)
)
