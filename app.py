
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"

st.set_page_config(
    page_title="MaintAI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_models():
    return joblib.load(MODELS_DIR / "maintai_all_models.joblib")

@st.cache_data
def load_metrics():
    path = REPORTS_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()

models = load_models()
metrics = load_metrics()

st.title("⚙️ MaintAI")
st.subheader("Explainable, Risk-Aware Predictive Maintenance")
st.caption(
    "Estimate machine failure risk from sensor readings and explore "
    "model performance. This is a research prototype, not a safety system."
)

with st.sidebar:
    st.header("Machine sensor inputs")

    machine_type = st.selectbox("Machine type", ["L", "M", "H"], index=1)
    air_temp = st.number_input(
        "Air temperature (K)", min_value=250.0, max_value=350.0,
        value=298.1, step=0.1
    )
    process_temp = st.number_input(
        "Process temperature (K)", min_value=250.0, max_value=400.0,
        value=308.6, step=0.1
    )
    speed = st.number_input(
        "Rotational speed (rpm)", min_value=0, max_value=5000,
        value=1551, step=10
    )
    torque = st.number_input(
        "Torque (Nm)", min_value=0.0, max_value=150.0,
        value=42.8, step=0.1
    )
    tool_wear = st.number_input(
        "Tool wear (minutes)", min_value=0, max_value=500,
        value=120, step=5
    )

    model_name = st.selectbox("Prediction model", list(models.keys()))
    threshold = st.slider(
        "Risk alert threshold (%)", 1, 99, 30
    ) / 100

sensor_data = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature": air_temp,
    "Process temperature": process_temp,
    "Rotational speed": speed,
    "Torque": torque,
    "Tool wear": tool_wear
}])

model = models[model_name]
feature_names = model.named_steps["preprocessor"].feature_names_in_
sensor_data = sensor_data[list(feature_names)]

probability = float(model.predict_proba(sensor_data)[0, 1])
predicted_class = int(probability >= threshold)

st.markdown("### Machine risk overview")
c1, c2, c3 = st.columns(3)

c1.metric("Estimated failure probability", f"{probability:.1%}")
c2.metric("Selected model", model_name)
c3.metric("Alert threshold", f"{threshold:.0%}")

if predicted_class:
    st.error("⚠️ Risk alert: the estimated probability meets or exceeds your alert threshold.")
else:
    st.success("No risk alert at the selected threshold. This does not guarantee the machine is safe.")

st.caption(
    "The threshold changes the alert classification, not the underlying estimated probability. "
    "Probabilities may not be calibrated and should be validated before operational use."
)

st.markdown("### Sensor readings")
st.dataframe(sensor_data, use_container_width=True, hide_index=True)

st.markdown("### Explainable maintenance guidance")
suggestions = []

if tool_wear >= 200:
    suggestions.append("Tool wear is high relative to this prototype's suggested review point; inspect the tool and maintenance log.")
elif tool_wear >= 150:
    suggestions.append("Tool wear is increasing; consider a routine tool inspection.")

if torque >= 60:
    suggestions.append("Torque is elevated; check load conditions and inspect for mechanical resistance.")
if speed > 2000:
    suggestions.append("Rotational speed is elevated; verify the machine's operating specification.")
if process_temp - air_temp < 5:
    suggestions.append("The temperature difference is small; verify the temperature sensors and process conditions.")
if probability >= threshold:
    suggestions.append("The model has raised a risk alert; inspect sensor trends and maintenance records before deciding on action.")

if not suggestions:
    suggestions.append("No simple sensor rule was triggered. Continue routine monitoring and follow the manufacturer's maintenance schedule.")

for item in suggestions:
    st.write(f"- {item}")

st.caption(
    "Guidance uses transparent prototype rules and the model estimate; it is not a causal explanation "
    "or a substitute for manufacturer limits and professional inspection."
)

st.markdown("### Model comparison")
if not metrics.empty and "Model" in metrics.columns:
    st.dataframe(
        metrics.round(3),
        use_container_width=True,
        hide_index=True
    )

    metric_columns = [
        c for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        if c in metrics.columns
    ]

    if metric_columns:
        chart_data = metrics.melt(
            id_vars="Model",
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Score"
        )
        fig = px.bar(
            chart_data,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            range_y=[0, 1],
            title="Evaluation metrics on the held-out test set"
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### How this prototype works")
st.write(
    "AI4I 2020 sensor data → preprocessing → Decision Tree / Random Forest → "
    "estimated failure probability → threshold-based alert → maintenance guidance."
)
st.caption(
    "Dataset: UCI AI4I 2020 Predictive Maintenance. "
    "Performance on this dataset does not guarantee performance on real machinery."
)
