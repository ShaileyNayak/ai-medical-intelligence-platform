import os

import plotly.express as px
import requests
import streamlit as st

from components.ui_helpers import render_prediction

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="AI Medical Intelligence", layout="wide")
st.title("AI Medical Intelligence Platform")
st.caption(
    "Chest X-ray Normal vs Pneumonia — Grad-CAM + LLM assistive report. "
    "Educational prototype only."
)

tab_predict, tab_history = st.tabs(["Predict", "History"])

with tab_predict:
    uploaded = st.file_uploader("Chest X-ray", type=["png", "jpg", "jpeg", "webp"])
    if uploaded and st.button("Run prediction", type="primary"):
        with st.spinner("Running inference → Grad-CAM → report…"):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                response = requests.post(f"{API_BASE}/api/predict", files=files, timeout=180)
                response.raise_for_status()
                result = response.json()
                render_prediction(result, api_base=API_BASE)
                fig = px.bar(
                    x=[result["prediction"]],
                    y=[result["confidence"] * 100],
                    labels={"x": "Class", "y": "Confidence %"},
                    title="Confidence",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab_history:
    if st.button("Refresh history"):
        st.rerun()
    try:
        hist = requests.get(f"{API_BASE}/api/history", timeout=30).json()
        st.write(f"Total: {hist.get('total', 0)}")
        st.dataframe(hist.get("items", []), use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not load history: {exc}")
