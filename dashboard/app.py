"""Interactive demo: set a machine's sensor values, see the failure risk and why.

Run with:  streamlit run dashboard/app.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

from src.bootstrap import ensure_model  # noqa: E402
from src.explain import make_explainer, shap_explanation  # noqa: E402
from src.features import build_x  # noqa: E402

REPORTS_DIR = ROOT / "reports"

# Diverging pair (blue = toward healthy, red = toward failure) and status colours
# from the validated reference palette. Text never takes these colours.
TOWARD_FAILURE = "#e34948"
TOWARD_HEALTHY = "#2a78d6"
LINE = "#2a78d6"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

PRESETS = {
    "Healthy machine": dict(type="M", air_temp_k=298.1, process_temp_k=308.6, rpm=1551, torque_nm=42.8, tool_wear_min=108),
    "Heat dissipation risk": dict(type="M", air_temp_k=302.6, process_temp_k=310.6, rpm=1320, torque_nm=48.0, tool_wear_min=60),
    "Power too low": dict(type="M", air_temp_k=299.0, process_temp_k=309.5, rpm=2800, torque_nm=9.0, tool_wear_min=80),
    "Power too high": dict(type="M", air_temp_k=299.0, process_temp_k=309.5, rpm=1250, torque_nm=72.0, tool_wear_min=80),
    "Overstrained worn tool": dict(type="L", air_temp_k=299.0, process_temp_k=309.5, rpm=1400, torque_nm=55.0, tool_wear_min=225),
}

# slider key -> (label, min, max, step, display format)
SENSORS = {
    "air_temp_k": ("Air temperature [K]", 295.0, 305.0, 0.1, "%.1f"),
    "process_temp_k": ("Process temperature [K]", 305.0, 314.0, 0.1, "%.1f"),
    "rpm": ("Rotational speed [rpm]", 1150, 2900, 10, "%d"),
    "torque_nm": ("Torque [Nm]", 3.0, 77.0, 0.5, "%.1f"),
    "tool_wear_min": ("Tool wear [min]", 0, 255, 1, "%d"),
}

FEATURE_LABELS = {
    "air_temp_k": "Air temperature", "process_temp_k": "Process temperature",
    "rpm": "Rotational speed", "torque_nm": "Torque", "tool_wear_min": "Tool wear",
    "temp_diff_k": "Temp. gap (process − air)", "power_w": "Mechanical power",
    "wear_torque": "Tool wear × torque", "type_L": "Type L", "type_M": "Type M", "type_H": "Type H",
}


@st.cache_resource(show_spinner="First start: downloading the data and rebuilding the model…")
def load_model(models_dir: str):
    models_dir = Path(models_dir)
    path = models_dir / "model.joblib"
    if not path.exists() and (models_dir / "metadata.json").exists():
        ensure_model(models_dir)  # fresh clone / cloud host: rebuild the champion
    if not path.exists():
        return None, None, None
    model = joblib.load(path)
    meta = json.loads((models_dir / "metadata.json").read_text())
    return model, meta, make_explainer(model)


def apply_preset():
    for k, v in PRESETS[st.session_state.preset].items():
        st.session_state[k] = v


def reading_from_state() -> dict:
    return {"type": st.session_state.type, **{k: float(st.session_state[k]) for k in SENSORS}}


def chart_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        hoverlabel=dict(font_size=13), showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def drivers_chart(exp_row) -> go.Figure:
    df = pd.DataFrame({
        "feature": [FEATURE_LABELS[f] for f in exp_row.feature_names],
        "value": exp_row.data, "shap": exp_row.values,
    })
    df = df[df.shap.abs() >= 0.005]  # hide negligible contributions
    df = df.loc[df.shap.abs().sort_values().index]
    colors = [TOWARD_FAILURE if s > 0 else TOWARD_HEALTHY for s in df.shap]
    fig = go.Figure(go.Bar(
        x=df.shap, y=df.feature, orientation="h", marker=dict(color=colors, cornerradius=4),
        customdata=np.stack([df.value], axis=1),
        hovertemplate="<b>%{y}</b><br>value: %{customdata[0]:,.1f}<br>"
                      "contribution: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="rgba(128,128,128,0.6)")
    fig.update_xaxes(title_text="← pushes toward healthy     |     pushes toward failure →",
                     title_font_size=12)
    return chart_layout(fig, height=max(220, 34 * len(df) + 60))


def what_if_chart(model, reading: dict, sensor: str, threshold: float) -> go.Figure:
    label, lo, hi, _, _ = SENSORS[sensor]
    grid = np.linspace(lo, hi, 120)
    rows = pd.DataFrame([reading] * len(grid)).assign(**{sensor: grid})
    if sensor == "air_temp_k":  # keep the process hotter than ambient
        rows["process_temp_k"] = np.maximum(rows["process_temp_k"], grid)
    proba = model.predict_proba(build_x(rows))[:, 1]
    current = model.predict_proba(build_x(pd.DataFrame([reading])))[0, 1]

    fig = go.Figure()
    fig.add_hrect(y0=threshold, y1=1.02, fillcolor=TOWARD_FAILURE, opacity=0.07, line_width=0)
    fig.add_trace(go.Scatter(
        x=grid, y=proba, mode="lines", line=dict(color=LINE, width=2),
        hovertemplate=f"{label}: %{{x:,.1f}}<br>failure probability: %{{y:.0%}}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_width=1, line_color="rgba(128,128,128,0.9)",
                  annotation_text=f"alarm threshold {threshold:.2f}", annotation_position="top left",
                  annotation_font_size=12)
    fig.add_trace(go.Scatter(
        x=[reading[sensor]], y=[current], mode="markers",
        marker=dict(size=11, color=LINE, line=dict(width=2, color="white")),
        hovertemplate="current setting<br>%{x:,.1f} → %{y:.0%}<extra></extra>",
    ))
    fig.update_yaxes(range=[-0.02, 1.02], tickformat=".0%", title_text="failure probability")
    fig.update_xaxes(title_text=label)
    fig.update_layout(hovermode="x unified")
    return chart_layout(fig, height=320)


def predictor_tab(model, meta, explainer):
    reading = reading_from_state()
    X = build_x(pd.DataFrame([reading]))
    proba = float(model.predict_proba(X)[0, 1])
    threshold = float(meta["threshold"])
    flagged = proba >= threshold
    exp_row = shap_explanation(model, X, explainer)[0]

    c1, c2, c3 = st.columns([1.1, 1, 1.4])
    c1.metric("Failure probability", f"{proba:.0%}")
    with c2:
        color = STATUS_CRITICAL if flagged else STATUS_GOOD
        icon, text = ("⚠", "Schedule inspection") if flagged else ("✓", "No action needed")
        st.markdown(
            f"<div style='margin-top:4px'><span style='font-size:0.875rem;opacity:0.7'>Decision</span><br>"
            f"<span style='display:inline-flex;align-items:center;gap:8px;margin-top:6px'>"
            f"<span style='width:22px;height:22px;border-radius:50%;background:{color};color:white;"
            f"display:inline-flex;align-items:center;justify-content:center;font-size:13px'>{icon}</span>"
            f"<span style='font-size:1.15rem;font-weight:600'>{text}</span></span></div>",
            unsafe_allow_html=True,
        )
    c3.markdown(
        f"<span style='font-size:0.875rem;opacity:0.7'>Why this threshold</span><br>"
        f"Alarm at **≥ {threshold:.2f}**: the cut-off with the lowest cost when a missed failure "
        f"costs 10× a false alarm.",
        unsafe_allow_html=True,
    )

    st.divider()
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("Why the model says this")
        st.caption("SHAP contribution of each input. Red bars push toward failure, blue toward healthy.")
        st.plotly_chart(drivers_chart(exp_row), width="stretch", config={"displayModeBar": False})
        derived = pd.DataFrame({
            "Derived feature": ["Temp. gap (process − air)", "Mechanical power", "Tool wear × torque"],
            "Value": [f"{X.temp_diff_k[0]:.1f} K", f"{X.power_w[0]:,.0f} W", f"{X.wear_torque[0]:,.0f} min·Nm"],
            "Risk when": ["small (heat builds up)", "too low or too high", "high (overstrain)"],
        })
        st.dataframe(derived, hide_index=True, width="stretch")
    with right:
        st.subheader("What if one sensor changes?")
        sensor = st.selectbox("Sweep this sensor, holding the others fixed",
                              list(SENSORS), index=4, format_func=lambda k: SENSORS[k][0])
        st.plotly_chart(what_if_chart(model, reading, sensor, threshold), width="stretch",
                        config={"displayModeBar": False})
        st.caption("The dot is the current setting. The shaded band is where the alarm fires.")


def model_card_tab(meta):
    st.subheader(f"Production model: {meta['registered_model']} v{meta['registered_version']}")
    run_id = meta.get("mlflow_run_id")
    st.caption(f"{meta['model']} · threshold {meta['threshold']:.2f}"
               + (f" · MLflow run {run_id[:8]}" if run_id else ""))
    tm = meta.get("test_metrics") or {}
    if tm:
        cols = st.columns(5)
        cols[0].metric("Recall", f"{tm['recall']:.0%}", help="Share of real failures caught")
        cols[1].metric("Precision", f"{tm['precision']:.0%}", help="Share of alarms that were real")
        cols[2].metric("PR-AUC", f"{tm['pr_auc']:.3f}")
        cols[3].metric("Failures caught", f"{tm['tp']} / {tm['tp'] + tm['fn']}")
        cols[4].metric("False alarms", f"{tm['fp']}", help="Out of 2,000 test machines")

    comp = REPORTS_DIR / "model_comparison.csv"
    if comp.exists():
        st.markdown("**All candidates** (selected on cross-validated PR-AUC, never on the test set)")
        df = pd.read_csv(comp)[["run", "threshold", "cv_pr_auc", "test_pr_auc", "test_recall",
                                "test_precision", "test_cost"]]
        st.dataframe(df.round(3), hide_index=True, width="stretch")

    left, right = st.columns(2)
    promo = REPORTS_DIR / "promotion.json"
    if promo.exists():
        p = json.loads(promo.read_text())
        left.markdown("**Last promotion decision**")
        left.write(f"{'Promoted' if p['promoted'] else 'Kept champion'}: {p['reason']} "
                   f"(quality gate: {p['quality_gate']})")
    drift = REPORTS_DIR / "drift_summary.json"
    if drift.exists():
        right.markdown("**Drift check**")
        for r in json.loads(drift.read_text()):
            verdict = "⚠ retrain" if r["retrain_recommended"] else "✓ ok"
            feats = ", ".join(r["drifted_features"]) or "none"
            right.write(f"{verdict} · **{r['batch']}** batch: {r['share_drifted']:.0%} drifted ({feats})")


def main():
    st.set_page_config(page_title="Machine Failure Risk", page_icon="⚙", layout="wide")
    model, meta, explainer = load_model(os.getenv("MODELS_DIR", str(ROOT / "models")))
    if model is None:
        st.error("No model found in models/. Run `make pipeline` first.")
        st.stop()

    if "preset" not in st.session_state:
        st.session_state.preset = "Overstrained worn tool"
        apply_preset()

    with st.sidebar:
        st.header("Machine reading")
        st.selectbox("Start from a scenario", list(PRESETS), key="preset", on_change=apply_preset)
        st.radio("Product quality", ["L", "M", "H"], key="type", horizontal=True,
                 format_func={"L": "Low", "M": "Medium", "H": "High"}.get)
        for key, (label, lo, hi, step, fmt) in SENSORS.items():
            st.slider(label, lo, hi, step=step, format=fmt, key=key)
        if st.session_state.process_temp_k < st.session_state.air_temp_k:
            st.warning("Process temperature is below air temperature; the API would reject this reading.")

    st.title("Machine failure risk")
    st.caption("AI4I 2020 milling-machine data (synthetic). Random Forest, class-weighted and tuned, "
               "with SHAP explanations. The same model and code as the FastAPI service.")
    tab1, tab2 = st.tabs(["Predict & explain", "Model card"])
    with tab1:
        predictor_tab(model, meta, explainer)
    with tab2:
        model_card_tab(meta)


main()
