from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_runtime import load_locked_artifacts, predict_patient


st.set_page_config(
    page_title="ALS Depressive-Symptom Risk Prediction Tool",
    page_icon="ALS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');
:root { --navy:#12385d; --blue:#1f5f8b; --teal:#2a7f79; --ink:#172b3a; --muted:#617485; --line:#dce6ed; --paper:#f5f8fa; --amber:#b56a1f; }
.stApp { background: linear-gradient(180deg,#edf3f7 0,#f8fafb 18rem,#f8fafb 100%); color:var(--ink); font-family:'Source Sans 3',sans-serif; }
.block-container { max-width:1180px; padding-top:2.2rem; padding-bottom:4rem; }
.hero { position:relative; overflow:hidden; padding:2.2rem 2.4rem; background:#fff; border:1px solid var(--line); border-top:5px solid var(--navy); box-shadow:0 14px 40px rgba(22,55,82,.08); }
.hero:after { content:''; position:absolute; width:240px; height:240px; right:-95px; top:-120px; border:28px solid rgba(42,127,121,.09); border-radius:50%; }
.eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.14em; font-size:.76rem; font-weight:700; margin-bottom:.55rem; }
.hero h1 { color:var(--navy); font-family:'Source Serif 4',serif; font-size:clamp(2rem,4vw,3.35rem); line-height:1.08; margin:.1rem 0 .65rem; }
.hero .subtitle { color:#335b78; font-family:'Source Serif 4',serif; font-size:1.18rem; margin:0; }
.hero .meta { color:var(--muted); max-width:780px; line-height:1.8; margin:1.1rem 0 0; }
.section-label { color:var(--teal); font-size:.76rem; letter-spacing:.13em; text-transform:uppercase; font-weight:700; margin-top:1.6rem; }
.section-title { color:var(--navy); font-family:'Source Serif 4',serif; font-size:1.55rem; margin:.25rem 0 .8rem; }
.info-strip { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--line); border:1px solid var(--line); margin:1rem 0 1.5rem; }
.info-strip > div { background:#fff; padding:1rem 1.1rem; }
.info-strip b { display:block; color:var(--navy); margin-bottom:.28rem; }
.info-strip span { color:var(--muted); font-size:.9rem; line-height:1.55; }
.result-card { background:#fff; border:1px solid var(--line); border-left:6px solid var(--teal); padding:1.35rem 1.45rem; box-shadow:0 10px 28px rgba(22,55,82,.07); }
.result-card.high { border-left-color:#c77a2a; }
.probability { color:var(--navy); font-family:'Source Serif 4',serif; font-size:3rem; line-height:1; font-weight:700; }
.risk-low,.risk-high { display:inline-block; margin-top:.6rem; padding:.36rem .72rem; border-radius:2px; font-weight:700; }
.risk-low { color:#17635e; background:#e6f3f1; }
.risk-high { color:#8a4b13; background:#fff0df; }
.micro-note { color:var(--muted); font-size:.82rem; line-height:1.65; }
.privacy { padding:1rem 1.2rem; border:1px solid #cfe1df; background:#f0f8f7; color:#315f5c; margin-top:1rem; }
div[data-baseweb='tab-list'] { gap:.4rem; border-bottom:1px solid var(--line); }
button[data-baseweb='tab'] { font-weight:700; padding:.8rem 1.25rem; }
div[data-testid='stForm'] { background:#fff; border:1px solid var(--line); padding:1rem 1.2rem 1.2rem; }
.stButton > button, .stFormSubmitButton > button { background:var(--navy); color:white; border:0; border-radius:2px; font-weight:700; min-height:3rem; }
.stButton > button:hover, .stFormSubmitButton > button:hover { background:var(--blue); color:white; }
div[data-testid='stMetric'] { background:#fff; border:1px solid var(--line); padding:.7rem 1rem; }
@media(max-width:760px){ .block-container{padding:1rem .8rem 3rem}.hero{padding:1.5rem 1.15rem}.info-strip{grid-template-columns:1fr}.probability{font-size:2.45rem} }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading locked models...")
def model_bank():
    return {
        "3_month": load_locked_artifacts("3_month"),
        "6_month": load_locked_artifacts("6_month"),
    }


def warning_messages(outcome: str, values: dict[str, float]) -> list[str]:
    ranges = {
        "alsfrsr": (0, 48, "ALSFRS-R"),
        "fss": (9, 63, "FSS"),
        "psqi": (0, 21, "PSQI"),
        "albumin": (30, 55, "Albumin"),
        "lymphocytes": (0.8, 4.0, "Lymphocyte count"),
        "neutrophils": (1.8, 7.5, "Neutrophil count"),
    }
    if outcome == "3_month":
        ranges["phase_angle"] = (3.0, 8.0, "Phase angle")
    else:
        ranges["age"] = (18, 85, "Age")
        ranges["hemoglobin"] = (100, 175, "Hemoglobin")
    return [f"{label} is outside the usual clinical range shown by this tool ({low}-{high}). Please verify the value and unit." for key, (low, high, label) in ranges.items() if values[key] < low or values[key] > high]


def number_field(label, key, value, min_value=None, max_value=None, step=1.0, help_text=None):
    return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step, help=help_text, key=key)


def collect_inputs(outcome: str) -> dict[str, float] | None:
    with st.form(f"form_{outcome}", clear_on_submit=False):
        st.markdown("#### Baseline variables")
        col1, col2 = st.columns(2)
        with col1:
            if outcome == "6_month":
                age = number_field("Age (years)", f"age_{outcome}", 55.0, 18.0, 110.0, 1.0)
            else:
                age = None
            alsfrsr = number_field("ALSFRS-R (0-48)", f"alsfrsr_{outcome}", 38.0, 0.0, 48.0, 1.0, "Higher scores indicate better functional status.")
            fss = number_field("FSS (9-63)", f"fss_{outcome}", 36.0, 9.0, 63.0, 1.0)
            psqi = number_field("PSQI (0-21)", f"psqi_{outcome}", 8.0, 0.0, 21.0, 1.0)
        with col2:
            albumin = number_field("Albumin (g/L)", f"albumin_{outcome}", 40.0, 5.0, 80.0, 0.1)
            if outcome == "3_month":
                phase_angle = number_field("Phase angle (degrees)", f"phase_{outcome}", 5.0, 0.1, 20.0, 0.1)
                hemoglobin = None
            else:
                phase_angle = None
                hemoglobin = number_field("Hemoglobin (g/L)", f"hb_{outcome}", 135.0, 30.0, 250.0, 1.0)
            lymphocytes = number_field("Lymphocyte count (x10^9/L)", f"lymph_{outcome}", 1.6, 0.01, 20.0, 0.01)
            neutrophils = number_field("Neutrophil count (x10^9/L)", f"neut_{outcome}", 3.8, 0.0, 50.0, 0.01)
        nlr = neutrophils / lymphocytes
        st.caption(f"The tool automatically calculates NLR = {nlr:.3f} and log1p(NLR) = {np.log1p(nlr):.3f}. Neutrophil count is used only to calculate NLR and is not entered as an independent model predictor.")
        submitted = st.form_submit_button("Calculate risk", width="stretch")
    if not submitted:
        return None
    values = {
        "alsfrsr": alsfrsr,
        "fss": fss,
        "psqi": psqi,
        "albumin": albumin,
        "lymphocytes": lymphocytes,
        "neutrophils": neutrophils,
    }
    if outcome == "3_month":
        values["phase_angle"] = phase_angle
    else:
        values["age"] = age
        values["hemoglobin"] = hemoglobin
    return {key: float(value) for key, value in values.items()}


def contribution_chart(table: pd.DataFrame, scale: str):
    plot = table.sort_values("Contribution")
    colors = ["#2b6f9f" if value < 0 else "#c77a2a" for value in plot["Contribution"]]
    fig = go.Figure(go.Bar(x=plot["Contribution"], y=plot["Feature"], orientation="h", marker_color=colors, text=[f"{v:+.3f}" for v in plot["Contribution"]], textposition="auto", cliponaxis=False))
    fig.add_vline(x=0, line_width=1, line_color="#718292")
    fig.update_layout(height=380, margin=dict(l=130, r=80, t=20, b=50), xaxis_title=scale, yaxis_title=None, paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Source Sans 3", size=13, color="#233746"), showlegend=False)
    fig.update_xaxes(gridcolor="#e6edf2", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def show_result(outcome: str, values: dict[str, float]):
    for message in warning_messages(outcome, values):
        st.warning(message)
    with st.spinner("Calculating the mean prediction from 20 imputation-specific models..."):
        result = predict_patient(outcome, values, model_bank()[outcome])
    month = "3 months" if outcome == "3_month" else "6 months"
    high_class = "high" if result.risk_label.startswith("Higher") else ""
    risk_class = "risk-high" if high_class else "risk-low"
    st.markdown(
        f"""
<div class="result-card {high_class}">
  <div class="eyebrow">{month} prediction result</div>
  <div class="probability">{result.probability:.3f}</div>
  <div class="micro-note">Arithmetic mean of predictions from 20 locked models</div>
  <span class="{risk_class}">{result.risk_label}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction horizon", month)
    c2.metric("Fixed threshold", f"{result.threshold:.3f}")
    c3.metric("Number of models", "20")
    progress = min(max(result.probability, 0.0), 1.0)
    st.progress(progress, text=f"Predicted probability {result.probability:.3f} | Threshold {result.threshold:.3f}")
    st.info("This result represents the model-predicted risk of depressive symptoms and is not a clinical diagnosis. For patients with higher predicted risk, further assessment using validated scales, clinical interviews, and professional evaluation should be considered.")

    st.markdown("##### Input summary")
    labels = {
        "age": "Age (years)", "alsfrsr": "ALSFRS-R", "fss": "FSS", "psqi": "PSQI",
        "albumin": "Albumin (g/L)", "phase_angle": "Phase angle (degrees)", "hemoglobin": "Hemoglobin (g/L)",
        "lymphocytes": "Lymphocyte count (x10^9/L)", "neutrophils": "Neutrophil count (x10^9/L)",
    }
    summary = pd.DataFrame({"Variable": [labels[key] for key in values], "Input value": [values[key] for key in values]})
    summary.loc[len(summary)] = ["NLR (automatically calculated)", values["neutrophils"] / values["lymphocytes"]]
    st.dataframe(summary, hide_index=True, width="stretch")

    st.markdown("##### Individual model explanation")
    contribution_chart(result.contributions, result.explanation_scale)
    st.caption("Positive orange values increase the model output, whereas negative blue values decrease it. Contributions are patient-level averages across 20 imputation-specific models.")
    if outcome == "6_month":
        st.caption(f"Maximum absolute TreeSHAP additivity error: {result.additivity_error:.2e} (prespecified acceptance criterion <=1x10^-4).")
    st.warning("Feature contributions describe model behavior only. They do not establish causal relationships with depressive symptoms or identify independent risk factors.")


st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Clinical research prediction interface</div>
  <h1>ALS Depressive-Symptom Risk Prediction Tool</h1>
  <p class="subtitle">Research interface for 3- and 6-month risk estimation</p>
  <p class="meta">This tool uses locked 3- and 6-month models to estimate depressive-symptom risk from baseline clinical, scale, and laboratory data in patients with ALS. It does not collect identifying information, store user inputs, or replace clinical diagnosis.</p>
</div>
<div class="info-strip">
  <div><b>Locked models</b><span>Uses 20 formally saved imputation-specific models without online training or tuning.</span></div>
  <div><b>Temporal evaluation</b><span>Performance was evaluated in later-in-time patients from a single center.</span></div>
  <div><b>Privacy first</b><span>Calculations occur in server memory without database storage or patient-data logging.</span></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Risk estimation</div><div class="section-title">Select a prediction horizon and enter baseline information</div>', unsafe_allow_html=True)
tab3, tab6 = st.tabs(["3-month risk", "6-month risk"])
with tab3:
    values3 = collect_inputs("3_month")
    if values3 is not None:
        show_result("3_month", values3)
with tab6:
    values6 = collect_inputs("6_month")
    if values6 is not None:
        show_result("6_month", values6)

st.markdown('<div class="section-label">Documentation</div><div class="section-title">Instructions, model information, and limitations</div>', unsafe_allow_html=True)
with st.expander("How to use this tool"):
    st.markdown("1. Select the 3- or 6-month prediction horizon.\n2. Enter baseline variables using the displayed units.\n3. Verify the automatically calculated NLR.\n4. Select **Calculate risk** to view the predicted probability, fixed threshold, risk category, and model explanation.\n\nDo not enter names, identification numbers, hospital numbers, telephone numbers, addresses, or other identifying information.")
with st.expander("Model information"):
    info = pd.DataFrame([
        ["3 months", "Logistic regression", 7, "0.350", "0.729", "0.614", "0.178"],
        ["6 months", "XGBoost", 8, "0.280", "0.768", "0.675", "0.201"],
    ], columns=["Horizon", "Algorithm", "Predictors", "Threshold", "Temporal-validation AUROC", "AUPRC", "Brier score"])
    st.dataframe(info, hide_index=True, width="stretch")
    st.caption("The models underwent single-center temporal evaluation but have not yet undergone independent multicenter external validation.")
with st.expander("Version information"):
    st.markdown("**Model version:** ALS-DSRP 1.0.0  \n**Model generation date:** 2026-07-14  \n**Web application version:** 1.0.1  \n**Last updated:** 2026-07-15")

st.markdown('<div class="privacy"><b>Privacy and security</b><br>All predictions are calculated in server memory. The application does not create a patient database, store user inputs, call third-party patient-data APIs, use Google Analytics, or transmit patient information to external analytics services. Inputs are not retained after the page is refreshed or the session ends.</div>', unsafe_allow_html=True)
st.caption("For research demonstration and risk stratification only. This tool does not replace psychiatric diagnosis, clinical interviews, or medical decision-making.")
