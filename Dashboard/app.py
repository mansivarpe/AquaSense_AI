
import requests
import pandas as pd
import streamlit as st

from datetime import datetime


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AquaSense | Water Monitoring",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "https://aquasense-ai-1-7t8u.onrender.com"


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(100, 160, 200, 0.08);
            border: 1px solid rgba(100, 160, 200, 0.25);
            padding: 16px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# API functions
# --------------------------------------------------

@st.cache_data(ttl=10, show_spinner=False)
def get_api_data(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def to_dataframe(data):
    if not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(
            df["created_at"], errors="coerce"
        )

    if "predicted_at" in df.columns:
        df["predicted_at"] = pd.to_datetime(
            df["predicted_at"], errors="coerce"
        )

    return df


def show_value(value, suffix=""):
    if value is None or pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.2f}{suffix}"
    except (ValueError, TypeError):
        return f"{value}{suffix}"


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("💧 AquaSense")
st.sidebar.caption("Water Level & Quality Monitoring")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Sensor Monitoring",
        "ML Anomaly Analysis",
        "Alert History",
    ],
)

st.sidebar.divider()
st.sidebar.write("**Backend API**")
st.sidebar.code(API_BASE_URL)

if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    get_api_data.clear()
    st.rerun()

st.sidebar.caption(
    f"Last dashboard refresh: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)


# --------------------------------------------------
# API health check
# --------------------------------------------------

health, health_error = get_api_data("/health")

if health_error:
    st.error(
        "Cannot connect to FastAPI. Start the backend first using:\n\n"
        "`python -m uvicorn Backend.main:app --reload`"
    )
    st.stop()

st.title("💧 Intelligent Water Monitoring Dashboard")
st.caption(
    "Sensor readings, water quality indicators, and "
    "machine-learning anomaly monitoring"
)

if health and health.get("database_status") == "connected":
    st.success("Backend connected · PostgreSQL connected")
else:
    st.warning("The backend responded, but database health is uncertain.")


# --------------------------------------------------
# Fetch common dashboard data
# --------------------------------------------------

summary, summary_error = get_api_data("/api/summary")

readings_data, readings_error = get_api_data(
    "/api/readings/latest",
    {"limit": 500},
)

alerts_data, alerts_error = get_api_data(
    "/api/alerts",
    {"limit": 500},
)

if summary_error:
    st.error(f"Could not load summary: {summary_error}")

if readings_error:
    st.error(f"Could not load sensor readings: {readings_error}")

if alerts_error:
    st.error(f"Could not load alerts: {alerts_error}")

readings = to_dataframe(readings_data) if readings_data is not None else pd.DataFrame()
alerts = to_dataframe(alerts_data) if alerts_data is not None else pd.DataFrame()


# --------------------------------------------------
# Overview
# --------------------------------------------------

if page == "Overview":

    st.subheader("System Overview")

    if summary:
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Sensor Readings",
            summary.get("total_readings", 0),
        )
        c2.metric(
            "ML Predictions",
            summary.get("total_predictions", 0),
        )
        c3.metric(
            "Total Alerts",
            summary.get("total_alerts", 0),
        )
        c4.metric(
            "Anomalies",
            summary.get("anomaly_count", 0),
        )
        c5.metric(
            "Normal Predictions",
            summary.get("normal_count", 0),
        )

    st.divider()

    if not readings.empty:
        latest = readings.iloc[0]

        st.subheader("Latest Sensor Readings")

        a, b, c, d, e = st.columns(5)

        a.metric(
            "Water Level",
            show_value(latest.get("water_level_percentage"), "%"),
        )
        b.metric(
            "Temperature",
            show_value(latest.get("temperature_c"), " °C"),
        )
        c.metric(
            "TDS",
            show_value(latest.get("tds_ppm"), " ppm"),
        )
        d.metric(
            "Turbidity",
            show_value(latest.get("turbidity_ntu"), " NTU"),
        )
        e.metric(
            "pH",
            show_value(latest.get("ph_value")),
        )

        st.caption(
            "Latest database record: "
            + str(latest.get("timestamp", "Not available"))
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Water Level Trend")

            chart_df = readings.dropna(
                subset=["timestamp", "water_level_percentage"]
            ).sort_values("timestamp")

            if not chart_df.empty:
                chart_df = chart_df.set_index("timestamp")
                st.line_chart(
                    chart_df["water_level_percentage"],
                    y_label="Water level (%)",
                )
            else:
                st.info("No valid water-level values available.")

        with right:
            st.subheader("Water Quality Sensor Trends")

            available = [
                col for col in [
                    "temperature_c",
                    "tds_ppm",
                    "turbidity_ntu",
                    "ph_value",
                ]
                if col in readings.columns
            ]

            selected = st.selectbox(
                "Select a sensor",
                available,
                format_func=lambda x: {
                    "temperature_c": "Temperature (°C)",
                    "tds_ppm": "TDS (ppm)",
                    "turbidity_ntu": "Turbidity (NTU)",
                    "ph_value": "pH",
                }.get(x, x),
            )

            if selected:
                sensor_df = readings.dropna(
                    subset=["timestamp", selected]
                ).sort_values("timestamp")

                if not sensor_df.empty:
                    st.line_chart(
                        sensor_df.set_index("timestamp")[selected],
                        y_label=selected,
                    )

        st.subheader("Recent Sensor Records")

        display_cols = [
            col for col in [
                "reading_id",
                "timestamp",
                "water_level_percentage",
                "temperature_c",
                "tds_ppm",
                "turbidity_ntu",
                "ph_value",
                "water_quality_label",
                "anomaly_label",
                "review_alert",
            ]
            if col in readings.columns
        ]

        st.dataframe(
            readings[display_cols].head(20),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No sensor readings were returned by the API.")

    st.caption(
        "ML anomaly flags indicate unusual sensor patterns; "
        "they do not independently confirm that water is unsafe."
    )


# --------------------------------------------------
# Sensor Monitoring
# --------------------------------------------------

elif page == "Sensor Monitoring":

    st.subheader("Sensor Monitoring")

    if readings.empty:
        st.info("No sensor readings are available.")
    else:
        sensor_options = {
            "Water Level (cm)": "water_level_cm",
            "Water Level (%)": "water_level_percentage",
            "Temperature (°C)": "temperature_c",
            "TDS (ppm)": "tds_ppm",
            "Turbidity (NTU)": "turbidity_ntu",
            "pH": "ph_value",
            "Distance (cm)": "distance_cm",
        }

        selected_label = st.selectbox(
            "Choose a sensor",
            list(sensor_options.keys()),
        )

        selected_col = sensor_options[selected_label]

        if selected_col in readings.columns:
            plot_df = readings.dropna(
                subset=["timestamp", selected_col]
            ).sort_values("timestamp")

            if not plot_df.empty:
                st.line_chart(
                    plot_df.set_index("timestamp")[selected_col],
                    y_label=selected_label,
                )

                st.metric(
                    f"Latest {selected_label}",
                    show_value(plot_df.iloc[-1][selected_col]),
                )

                st.write("Sensor Data")

                st.dataframe(
                    plot_df[
                        ["timestamp", selected_col]
                    ].sort_values("timestamp", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No valid values for this sensor.")

        st.subheader("Device Status")

        status_cols = [
            col for col in [
                "timestamp",
                "pump_status",
                "buzzer_status",
                "led_status",
                "lcd_status",
            ]
            if col in readings.columns
        ]

        if status_cols:
            st.dataframe(
                readings[status_cols].head(50),
                use_container_width=True,
                hide_index=True,
            )


# --------------------------------------------------
# ML Anomaly Analysis
# --------------------------------------------------

elif page == "ML Anomaly Analysis":

    st.subheader("Machine Learning Anomaly Analysis")

    if readings.empty:
        st.info("No prediction records are available.")
    elif "anomaly_label" not in readings.columns:
        st.warning("The API response does not contain anomaly labels.")
    else:
        labels = readings["anomaly_label"].fillna("Unknown").astype(str)

        label_counts = labels.value_counts().rename_axis(
            "Prediction"
        ).to_frame("Records")

        left, right = st.columns(2)

        with left:
            st.subheader("Prediction Distribution")
            st.bar_chart(label_counts)

        with right:
            if "anomaly_score" in readings.columns:
                st.subheader("Anomaly Scores")

                score_df = readings.dropna(
                    subset=["timestamp", "anomaly_score"]
                ).sort_values("timestamp")

                if not score_df.empty:
                    st.line_chart(
                        score_df.set_index("timestamp")["anomaly_score"],
                        y_label="Anomaly score",
                    )
                else:
                    st.info("No anomaly scores are available.")

        anomaly_labels = labels.str.lower().isin(
            ["anomaly", "1", "-1", "true"]
        )

        st.subheader("Records Flagged for Review")

        anomalies = readings[anomaly_labels].copy()

        if not anomalies.empty:
            review_cols = [
                col for col in [
                    "reading_id",
                    "timestamp",
                    "water_level_percentage",
                    "temperature_c",
                    "tds_ppm",
                    "turbidity_ntu",
                    "ph_value",
                    "anomaly_label",
                    "anomaly_score",
                    "review_alert",
                ]
                if col in anomalies.columns
            ]

            st.dataframe(
                anomalies[review_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No records are currently flagged as anomalies.")

        st.caption(
            "These results are generated by the trained anomaly model. "
            "Investigate sensor calibration and readings before "
            "concluding that a fault or water-quality problem exists."
        )


# --------------------------------------------------
# Alert History
# --------------------------------------------------

elif page == "Alert History":

    st.subheader("Alert History")

    if alerts.empty:
        st.info("No alerts were returned by the API.")
    else:
        if "status" in alerts.columns:
            statuses = sorted(
                alerts["status"].dropna().astype(str).unique().tolist()
            )

            selected_statuses = st.multiselect(
                "Filter by status",
                statuses,
                default=statuses,
            )

            filtered_alerts = alerts[
                alerts["status"].astype(str).isin(selected_statuses)
            ]
        else:
            filtered_alerts = alerts

        st.metric("Displayed Alerts", len(filtered_alerts))

        alert_cols = [
            col for col in [
                "alert_id",
                "reading_id",
                "alert_type",
                "message",
                "status",
                "created_at",
                "reading_timestamp",
                "water_level_percentage",
                "water_quality_label",
            ]
            if col in filtered_alerts.columns
        ]

        st.dataframe(
            filtered_alerts[alert_cols],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = filtered_alerts[alert_cols].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download alerts as CSV",
            data=csv_data,
            file_name="water_monitoring_alerts.csv",
            mime="text/csv",
        )
