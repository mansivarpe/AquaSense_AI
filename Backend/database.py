
from pathlib import Path
from getpass import getpass

import pandas as pd
from sqlalchemy import create_engine, text, URL

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "Data" / "Processed" / "prediction_report.csv"

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Prediction report not found: {INPUT_FILE}\n"
        "Run python src/predict.py first."
    )

# PostgreSQL connection details
db_user = input("PostgreSQL username [postgres]: ").strip() or "postgres"
db_password = getpass("Enter PostgreSQL password: ")

db_url = URL.create(
    drivername="postgresql+psycopg",
    username=db_user,
    password=db_password,
    host="localhost",
    port=5432,
    database="Mini_Project_02",
)

engine = create_engine(db_url)

try:
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    # Required CSV columns: no Water_Level_cm
    reading_csv_columns = [
        "Timestamp",
        "Tank_Height_cm",
        "Distance_cm",
        "Water_Level_Percentage",
        "Temperature_C",
        "TDS_PPM",
        "Turbidity_NTU",
        "pH_Value",
        "Potentiometer_Value",
        "Pump_Status",
        "Buzzer_Status",
        "LED_Status",
        "LCD_Status",
        "Water_Quality_Label",
        "Fault_Label",
        "Water_Level_Change_cm_per_min",
    ]

    prediction_columns = [
        "Anomaly_Result",
        "Anomaly_Score",
        "Anomaly_Label",
        "Review_Alert",
    ]

    required_columns = reading_csv_columns + prediction_columns
    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in prediction report: {missing}"
        )

    if df.empty:
        raise ValueError("Prediction report contains no records.")

    # Parse timestamps
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )

    def clean_value(value):
        """Convert pandas/numpy values to database-compatible values."""
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if hasattr(value, "item"):
            return value.item()
        return value

    # Map PostgreSQL column names to CSV column names
    reading_columns = {
        "timestamp": "Timestamp",
        "tank_height_cm": "Tank_Height_cm",
        "distance_cm": "Distance_cm",
        "water_level_percentage": "Water_Level_Percentage",
        "temperature_c": "Temperature_C",
        "tds_ppm": "TDS_PPM",
        "turbidity_ntu": "Turbidity_NTU",
        "ph_value": "pH_Value",
        "potentiometer_value": "Potentiometer_Value",
        "pump_status": "Pump_Status",
        "buzzer_status": "Buzzer_Status",
        "led_status": "LED_Status",
        "lcd_status": "LCD_Status",
        "water_quality_label": "Water_Quality_Label",
        "fault_label": "Fault_Label",
        "water_level_change_cm_per_min":
            "Water_Level_Change_cm_per_min",
    }

    with engine.begin() as conn:
        # Verify required tables exist
        for table in (
            "sensor_readings",
            "ml_predictions",
            "alerts",
            "pump_events",
        ):
            exists = conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name = :table_name
                    )
                """),
                {"table_name": table},
            ).scalar_one()

            if not exists:
                raise RuntimeError(
                    f"Table '{table}' does not exist. "
                    "Create the project tables in pgAdmin first."
                )

        # Prevent accidental duplicate imports
        existing = conn.execute(
            text("SELECT COUNT(*) FROM sensor_readings")
        ).scalar_one()

        if existing:
            raise RuntimeError(
                f"Database already contains {existing} readings. "
                "Import stopped to prevent duplicates."
            )

        for index, row in df.iterrows():
            reading = {
                "source_row_number": index + 1,
                **{
                    db_col: clean_value(row[csv_col])
                    for db_col, csv_col in reading_columns.items()
                },
            }

            # Insert sensor readings (no water_level_cm)
            reading_id = conn.execute(
                text("""
                    INSERT INTO sensor_readings (
                        source_row_number,
                        timestamp,
                        tank_height_cm,
                        distance_cm,
                        water_level_percentage,
                        temperature_c,
                        tds_ppm,
                        turbidity_ntu,
                        ph_value,
                        potentiometer_value,
                        pump_status,
                        buzzer_status,
                        led_status,
                        lcd_status,
                        water_quality_label,
                        fault_label,
                        water_level_change_cm_per_min
                    )
                    VALUES (
                        :source_row_number,
                        :timestamp,
                        :tank_height_cm,
                        :distance_cm,
                        :water_level_percentage,
                        :temperature_c,
                        :tds_ppm,
                        :turbidity_ntu,
                        :ph_value,
                        :potentiometer_value,
                        :pump_status,
                        :buzzer_status,
                        :led_status,
                        :lcd_status,
                        :water_quality_label,
                        :fault_label,
                        :water_level_change_cm_per_min
                    )
                    RETURNING reading_id
                """),
                reading,
            ).scalar_one()

            # Insert ML prediction
            result = clean_value(row["Anomaly_Result"])
            score = clean_value(row["Anomaly_Score"])
            label = clean_value(row["Anomaly_Label"])
            alert = clean_value(row["Review_Alert"])

            conn.execute(
                text("""
                    INSERT INTO ml_predictions (
                        reading_id,
                        anomaly_result,
                        anomaly_score,
                        anomaly_label,
                        review_alert
                    )
                    VALUES (
                        :reading_id,
                        :result,
                        :score,
                        :label,
                        :alert
                    )
                """),
                {
                    "reading_id": reading_id,
                    "result": result,
                    "score": score,
                    "label": label,
                    "alert": alert,
                },
            )

            # Create alert for model-flagged anomalies
            if label == "Anomaly":
                conn.execute(
                    text("""
                        INSERT INTO alerts (
                            reading_id,
                            alert_type,
                            message
                        )
                        VALUES (:reading_id, :alert_type, :message)
                    """),
                    {
                        "reading_id": reading_id,
                        "alert_type": "ML Anomaly",
                        "message": (
                            "Unusual sensor reading flagged by the "
                            "Isolation Forest model. Review required."
                        ),
                    },
                )

            # Record pump status events
            pump_status = clean_value(row["Pump_Status"])

            if pump_status is not None:
                conn.execute(
                    text("""
                        INSERT INTO pump_events (
                            reading_id,
                            pump_status,
                            event_time
                        )
                        VALUES (
                            :reading_id,
                            :pump_status,
                            :event_time
                        )
                    """),
                    {
                        "reading_id": reading_id,
                        "pump_status": pump_status,
                        "event_time": reading["timestamp"],
                    },
                )

    print("\nDATABASE IMPORT COMPLETED")
    print("-" * 40)
    print("CSV records imported:", len(df))
    print("Sensor readings:", len(df))
    print("ML predictions:", len(df))
    print(
        "Anomalies flagged:",
        int((df["Anomaly_Label"] == "Anomaly").sum()),
    )
    print(
        "Normal predictions:",
        int((df["Anomaly_Label"] == "Normal").sum()),
    )
    print("Database: Mini_Project_02")

finally:
    engine.dispose()