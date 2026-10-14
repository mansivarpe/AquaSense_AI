
import os
from contextlib import asynccontextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Query


# --------------------------------------------------
# 1. PostgreSQL configuration
# --------------------------------------------------

DB_CONFIG = {
    "host": os.getenv("WATER_DB_HOST", "localhost"),
    "port": int(os.getenv("WATER_DB_PORT", "5432")),
    "dbname": os.getenv("WATER_DB_NAME", "Mini_Project_02"),
    "user": os.getenv("WATER_DB_USER", "postgres"),
    "password": os.getenv("WATER_DB_PASSWORD", ""),
    "connect_timeout": 5,
}


# --------------------------------------------------
# 2. Database connection
# --------------------------------------------------

def get_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except psycopg2.Error as exc:
        print(f"Database connection error: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to PostgreSQL. Check database settings.",
        ) from exc


def fetch_all(query, params=None):
    """Execute a SELECT query and return JSON-friendly dictionaries."""
    connection = get_connection()

    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params or ())
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except psycopg2.Error as exc:
        print(f"Database query error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Database query failed. Check the backend terminal.",
        ) from exc
    finally:
        connection.close()


def fetch_one(query, params=None):
    """Execute a SELECT query and return one dictionary."""
    connection = get_connection()

    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params or ())
            row = cursor.fetchone()
            return dict(row) if row else None
    except psycopg2.Error as exc:
        print(f"Database query error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Database query failed. Check the backend terminal.",
        ) from exc
    finally:
        connection.close()


# --------------------------------------------------
# 3. FastAPI application
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Water Monitoring API starting...")
    print(f"Configured database: {DB_CONFIG['dbname']}")
    yield
    print("Water Monitoring API stopped.")


app = FastAPI(
    title="Water Level and Quality Monitoring API",
    description=(
        "API for sensor readings, water quality monitoring, "
        "ML anomaly predictions, and alerts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# 4. Root endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Water Level and Quality Monitoring API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


# --------------------------------------------------
# 5. Health check
# --------------------------------------------------

@app.get("/health")
def health():
    row = fetch_one("SELECT 1 AS database_connection")

    return {
        "api_status": "healthy",
        "database_status": (
            "connected" if row and row["database_connection"] == 1
            else "unknown"
        ),
        "database": DB_CONFIG["dbname"],
    }


# --------------------------------------------------
# 6. Dashboard summary
# --------------------------------------------------

@app.get("/api/summary")
def get_summary():
    query = """
        SELECT
            (SELECT COUNT(*) FROM sensor_readings) AS total_readings,
            (SELECT COUNT(*) FROM ml_predictions) AS total_predictions,
            (SELECT COUNT(*) FROM alerts) AS total_alerts,
            (
                SELECT COUNT(*)
                FROM ml_predictions
                WHERE LOWER(anomaly_label) = 'anomaly'
            ) AS anomaly_count,
            (
                SELECT COUNT(*)
                FROM ml_predictions
                WHERE LOWER(anomaly_label) = 'normal'
            ) AS normal_count
    """

    result = fetch_one(query)

    if result is None:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve database summary.",
        )

    return result


# --------------------------------------------------
# 7. Latest sensor readings + ML predictions
# --------------------------------------------------

@app.get("/api/readings/latest")
def get_latest_readings(
    limit: int = Query(default=20, ge=1, le=500)
):
    query = """
        SELECT
            sr.reading_id,
            sr.timestamp,
            sr.tank_height_cm,
            sr.distance_cm,
            sr.water_level_percentage,
            sr.temperature_c,
            sr.tds_ppm,
            sr.turbidity_ntu,
            sr.ph_value,
            sr.potentiometer_value,
            sr.pump_status,
            sr.buzzer_status,
            sr.led_status,
            sr.lcd_status,
            sr.water_quality_label,
            sr.fault_label,
            sr.water_level_change_cm_per_min,
            mp.prediction_id,
            mp.anomaly_result,
            mp.anomaly_score,
            mp.anomaly_label,
            mp.review_alert,
            mp.model_name,
            mp.predicted_at
        FROM sensor_readings AS sr
        LEFT JOIN ml_predictions AS mp
            ON sr.reading_id = mp.reading_id
        ORDER BY sr.timestamp DESC, sr.reading_id DESC
        LIMIT %s
    """

    return fetch_all(query, (limit,))


# --------------------------------------------------
# 8. Sensor reading history
# --------------------------------------------------

@app.get("/api/readings/history")
def get_reading_history(
    limit: int = Query(default=100, ge=1, le=5000)
):
    query = """
        SELECT
            reading_id,
            timestamp,
            water_level_cm,
            water_level_percentage,
            temperature_c,
            tds_ppm,
            turbidity_ntu,
            ph_value,
            pump_status,
            water_quality_label,
            fault_label
        FROM sensor_readings
        ORDER BY timestamp DESC, reading_id DESC
        LIMIT %s
    """

    return fetch_all(query, (limit,))


# --------------------------------------------------
# 9. Alerts
# --------------------------------------------------

@app.get("/api/alerts")
def get_alerts(
    limit: int = Query(default=100, ge=1, le=5000)
):
    query = """
        SELECT
            a.alert_id,
            a.reading_id,
            a.alert_type,
            a.message,
            a.status,
            a.created_at,
            sr.timestamp AS reading_timestamp,
            sr.water_level_percentage,
            sr.water_quality_label
        FROM alerts AS a
        LEFT JOIN sensor_readings AS sr
            ON a.reading_id = sr.reading_id
        ORDER BY a.created_at DESC, a.alert_id DESC
        LIMIT %s
    """

    return fetch_all(query, (limit,))
