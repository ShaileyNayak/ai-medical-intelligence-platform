import streamlit as st


def render_prediction(result: dict, api_base: str = "") -> None:
    st.subheader("Prediction")
    st.write(f"**Label:** {result.get('predicted_label', '—')}")
    conf = result.get("confidence_score")
    if conf is not None:
        st.write(f"**Confidence:** {conf * 100:.1f}%")
    st.write(f"**Model:** {result.get('model_version', '—')}")

    cols = st.columns(2)
    image_url = result.get("image_url")
    heatmap_url = result.get("heatmap_url")
    if image_url:
        cols[0].image(f"{api_base}{image_url}", caption="Original")
    if heatmap_url:
        cols[1].image(f"{api_base}{heatmap_url}", caption="Grad-CAM")

    report = result.get("llm_report")
    if report:
        st.subheader("LLM Assistive Report")
        st.write(report)
