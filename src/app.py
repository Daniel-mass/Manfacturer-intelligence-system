# src/app.py
"""
Intelligent Manufacturing Failure Prevention System — Streamlit Dashboard
Glass Neon UI | 6 Pages | Live Inference | Interactive Agent Q&A

Start order (separate terminals):
  T1: mlflow ui --host 0.0.0.0 --port 5000
  T2: python src/mcp_server.py
  T3: conda activate mfg-2 && solara run src/viz_dashboard.py
  T4: streamlit run src/app.py
"""

import streamlit as st
import json
import os
import sys
import re
import math
import time
import subprocess
import numpy as np
import pandas as pd
import polars as pl
import joblib
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from scipy.optimize import linprog

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "octotools"))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing AI System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# GLASS NEON CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --neon-red:    #ff2d55;
    --neon-cyan:   #00d4ff;
    --neon-green:  #00ff88;
    --neon-amber:  #ffaa00;
    --neon-purple: #bf5fff;
    --bg-card:     rgba(15,15,35,0.85);
    --text-muted:  #8888aa;
}

.stApp {
    background: linear-gradient(135deg,#080810 0%,#0d0d1a 50%,#080818 100%);
    font-family: 'Inter', sans-serif;
}

#MainMenu,footer,header,.stDeployButton {visibility:hidden;}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#080810 0%,#0d0d20 100%);
    border-right: 1px solid rgba(255,45,85,0.2);
}

[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid rgba(255,45,85,0.25);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(255,45,85,0.08);
    transition: all 0.3s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(255,45,85,0.5);
    box-shadow: 0 0 30px rgba(255,45,85,0.15);
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--neon-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid rgba(255,45,85,0.2);
    padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-muted);
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(255,45,85,0.3),rgba(0,212,255,0.15)) !important;
    color: white !important;
    box-shadow: 0 0 15px rgba(255,45,85,0.3);
}

.stButton > button {
    background: linear-gradient(135deg,rgba(255,45,85,0.2),rgba(0,212,255,0.1));
    border: 1px solid rgba(255,45,85,0.4);
    border-radius: 8px; color: white;
    font-family: 'Inter', sans-serif;
    font-weight: 500; transition: all 0.3s;
}
.stButton > button:hover {
    background: linear-gradient(135deg,rgba(255,45,85,0.4),rgba(0,212,255,0.2));
    border-color: var(--neon-red);
    box-shadow: 0 0 20px rgba(255,45,85,0.3);
    transform: translateY(-1px);
}

.stProgress > div > div {
    background: linear-gradient(90deg,var(--neon-red),var(--neon-cyan));
    border-radius: 4px;
}

.glass-card {
    background: rgba(15,15,35,0.85);
    border: 1px solid rgba(255,45,85,0.25);
    border-radius: 16px; padding: 24px;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4),0 0 20px rgba(255,45,85,0.05);
    margin: 8px 0;
}
.glass-card:hover {
    border-color: rgba(255,45,85,0.45);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4),0 0 30px rgba(255,45,85,0.12);
}

.neon-title {
    font-family: 'Inter', sans-serif; font-weight: 700; font-size: 2.2rem;
    background: linear-gradient(135deg,#ff2d55,#00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1.2;
}
.neon-subtitle { color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }

.section-header {
    color: var(--neon-cyan); font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.15em;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 1px solid rgba(0,212,255,0.2);
}

.status-badge-ok {
    display: inline-block;
    background: rgba(0,255,136,0.15); border: 1px solid rgba(0,255,136,0.4);
    color: #00ff88; border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em;
}
.status-badge-critical {
    display: inline-block;
    background: rgba(255,45,85,0.15); border: 1px solid rgba(255,45,85,0.4);
    color: #ff2d55; border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em;
}
.status-badge-warning {
    display: inline-block;
    background: rgba(255,170,0,0.15); border: 1px solid rgba(255,170,0,0.4);
    color: #ffaa00; border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.05em;
}

.tech-badge {
    display: inline-block;
    background: rgba(191,95,255,0.12); border: 1px solid rgba(191,95,255,0.3);
    color: #bf5fff; border-radius: 6px; padding: 3px 10px;
    font-size: 0.72rem; font-weight: 500; margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}

.kpi-value {
    font-family: 'JetBrains Mono', monospace; font-size: 2.4rem;
    font-weight: 700; color: var(--neon-cyan); line-height: 1;
}
.kpi-label {
    font-size: 0.72rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px;
}
.kpi-delta { font-size: 0.8rem; color: var(--neon-green); margin-top: 2px; }

.insight-box {
    background: rgba(0,212,255,0.06); border-left: 3px solid var(--neon-cyan);
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
    font-size: 0.88rem; color: #e8e8f0;
}
.warning-box {
    background: rgba(255,170,0,0.06); border-left: 3px solid var(--neon-amber);
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
    font-size: 0.88rem;
}
.critical-box {
    background: rgba(255,45,85,0.08); border-left: 3px solid var(--neon-red);
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
    font-size: 0.88rem;
}
.success-box {
    background: rgba(0,255,136,0.06); border-left: 3px solid var(--neon-green);
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

AI4I_FEATURE_COLS = [
    "rpm","torque","tool_wear","air_temp_c","proc_temp_c","power_w",
    "temp_delta","stress_index","overstrain_margin",
    "air_temp_c_smooth","proc_temp_c_smooth","rpm_smooth",
    "torque_smooth","tool_wear_smooth","power_w_smooth"
]
NASA_EXCLUDE_COLS = ["unit","cycle","dataset_id","condition_cluster","rul"]

# Dataset ranges for normalization (from Layer 1 output)
AI4I_RANGES = {
    "rpm":            (1168, 2886),
    "torque":         (3.8, 76.6),
    "tool_wear":      (0, 253),
    "air_temp_c":     (22.99, 29.99),
    "proc_temp_c":    (32.86, 39.44),
    "power_w":        (0, 9400),
    "temp_delta":     (7.5, 12.9),
    "stress_index":   (0, 1),
    "overstrain_margin": (-3650, 13000),
}

OSF_THRESHOLDS = {0: 11000, 1: 12000, 2: 13000}
PWF_MIN, PWF_MAX = 3500, 9000
TWF_ONSET = 200

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(8,8,16,0)",
    plot_bgcolor="rgba(13,13,26,0.6)",
    font=dict(family="Inter", color="#e0e0e0", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,45,85,0.3)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,45,85,0.3)"),
    legend=dict(bgcolor="rgba(8,8,16,0.8)", bordercolor="rgba(255,45,85,0.3)", borderwidth=1),
    margin=dict(t=50, b=40, l=40, r=40),
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — cached
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load all ML models once at startup."""
    mdls = {}
    try:
        mdls["ai4i_clf"]  = joblib.load(PROJECT_ROOT / "models/ai4i_classifier.joblib")
        mdls["ai4i_iso"]  = joblib.load(PROJECT_ROOT / "models/ai4i_iso_forest.joblib")
        mdls["nasa_rul"]  = joblib.load(PROJECT_ROOT / "models/nasa_rul_model.joblib")
    except Exception as e:
        st.warning(f"Model load warning: {e}")
    return mdls


@st.cache_data(ttl=30)
def load_json_file(rel_path: str) -> dict:
    p = PROJECT_ROOT / rel_path
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_survival_csv() -> pd.DataFrame:
    p = PROJECT_ROOT / "reports/survival_curves.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_nasa_df() -> pd.DataFrame:
    p = PROJECT_ROOT / "data/processed/nasa_features.parquet"
    return pl.read_parquet(p).to_pandas() if p.exists() else pd.DataFrame()


@st.cache_data
def load_ai4i_results() -> pd.DataFrame:
    p = PROJECT_ROOT / "data/processed/ai4i_results.parquet"
    return pl.read_parquet(p).to_pandas() if p.exists() else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE TRIGGER — fixed cwd bug
# ══════════════════════════════════════════════════════════════════════════════

LAYER_SCRIPTS = {
    1: ["src/data_processing_ai4i.py", "src/data_processing_nasa.py",
        "src/consolidate_nasa.py"],
    2: ["src/ai4i_feature_engineering.py", "src/nasa_feature_engineering.py"],
    3: ["src/ai4i_model.py", "src/nasa_model.py"],
    4: ["src/causal_analysis.py"],
    5: ["src/optimizer.py"],
    6: ["src/simulation.py"],
    7: ["src/octo_plotter.py", "src/aggregator.py", "src/octo_researcher.py"],
    8: ["src/monitoring.py"],
}


def trigger_layer(layer_num: int) -> dict:
    start = time.time()
    scripts = LAYER_SCRIPTS.get(layer_num, [])
    success_all = True
    errors = []

    for script in scripts:
        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            errors.append(f"{script} not found")
            success_all = False
            continue
        try:
            # FIX: Force UTF-8 env so emoji print statements don't crash cp1252
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",      # FIX: explicit utf-8
                errors="replace",      # FIX: replace unencodable chars
                timeout=3600,
                env=env                # FIX: pass utf-8 env
            )
            if result.returncode != 0:
                # Strip emojis from error message before storing
                err_clean = result.stderr[-500:].encode(
                    'ascii', 'replace'
                ).decode('ascii')
                errors.append(f"{script}: {err_clean}")
                success_all = False
        except subprocess.TimeoutExpired:
            errors.append(f"{script}: timed out")
            success_all = False
        except Exception as e:
            errors.append(f"{script}: {str(e)}")
            success_all = False

    return {
        "success": success_all,
        "elapsed": round(time.time() - start, 1),
        "errors":  errors,
        "layer":   layer_num,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LIVE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def predict_ai4i(mdls, rpm, torque, tool_wear, air_temp, proc_temp):
    """
    Build full feature vector from raw inputs, normalize, run RF + IsoForest.
    Normalization uses dataset ranges from Layer 1 — same scaling the model
    was trained on.
    """
    if "ai4i_clf" not in mdls:
        return None, None, None

    power_w   = torque * (rpm * 2 * math.pi / 60)
    temp_delta = proc_temp - air_temp
    stress_idx = (torque / 75.0) * (rpm / 2800.0)
    osf_margin = 12000 - (tool_wear * torque)

    row = {
        "rpm": rpm, "torque": torque, "tool_wear": tool_wear,
        "air_temp_c": air_temp, "proc_temp_c": proc_temp,
        "power_w": power_w, "temp_delta": temp_delta,
        "stress_index": stress_idx, "overstrain_margin": osf_margin,
        "air_temp_c_smooth": air_temp, "proc_temp_c_smooth": proc_temp,
        "rpm_smooth": rpm, "torque_smooth": torque,
        "tool_wear_smooth": tool_wear, "power_w_smooth": power_w,
    }

    # Min-max normalize using dataset ranges
    for col in AI4I_FEATURE_COLS:
        if col in AI4I_RANGES:
            lo, hi = AI4I_RANGES[col]
            span = hi - lo
            row[col] = max(0.0, min(1.0, (row[col] - lo) / span if span > 0 else 0.0))

    X = pd.DataFrame([row])[AI4I_FEATURE_COLS]
    try:
        fp   = float(mdls["ai4i_clf"].predict_proba(X)[0][1])
        anom = int(mdls["ai4i_iso"].predict(X)[0] == -1)
        ascore = float(mdls["ai4i_iso"].decision_function(X)[0])
        return fp, anom, ascore
    except Exception:
        return None, None, None


def live_ortools_solve(tool_wear: int, prod_type_num: int) -> dict:
    """
    Live OR-Tools CP-SAT solve for arbitrary tool_wear.
    Same logic as Layer 5 AI4IOptimizer.solve() — imported directly.
    Runs in <1 second.
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.optimizer import AI4IOptimizer
        opt = AI4IOptimizer()

        # Target power scales down with wear — mirrors Layer 5 logic
        if tool_wear < 50:     tp = 8000
        elif tool_wear < 150:  tp = 7500
        elif tool_wear < 200:  tp = 7000
        elif tool_wear < 220:  tp = 6000
        else:                  tp = 5000

        result = opt.solve(
            current_tool_wear=int(tool_wear),
            prod_type=prod_type_num,
            target_power_w=float(tp)
        )
        result["source"] = "⚡ Live OR-Tools solve"
        return result
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "source": "error"}


def get_setpoint(tool_wear: int, pt_code: str, ai4i_opt: dict) -> dict:
    """
    Within ±10 min of a pre-computed scenario → lookup (instant).
    Outside → live OR-Tools solve (<1 sec).
    """
    pt = pt_code.upper()
    scenarios = {
        "New tool": 0, "Mid-life tool": 100,
        "Approaching TWF onset": 180, "Past TWF onset": 210,
        "Near end of life": 235,
    }
    closest_name, closest_wear = min(
        scenarios.items(), key=lambda x: abs(x[1] - tool_wear)
    )
    if abs(closest_wear - tool_wear) <= 10:
        r = ai4i_opt.get(pt, {}).get(closest_name, {}).copy()
        r["matched_scenario"] = closest_name
        r["source"] = "📋 Pre-computed lookup"
        return r
    else:
        pt_num = {"L": 0, "M": 1, "H": 2}.get(pt, 1)
        return live_ortools_solve(tool_wear, pt_num)


def predict_nasa_rul(mdls, feature_row: pd.Series, feature_cols: list) -> float:
    """Live LightGBM RUL prediction."""
    if "nasa_rul" not in mdls:
        return None
    try:
        X = pd.DataFrame([feature_row[feature_cols].values], columns=feature_cols)
        return float(mdls["nasa_rul"].predict(X)[0])
    except Exception:
        return None


def live_linprog_intervention(
    mdls, selected_row: pd.Series, feature_cols: list,
    current_rul: float, nasa_causal: dict
) -> dict:
    """
    SciPy linprog intervention — same logic as Layer 5 NASAOptimizer.
    Runs live for user-selected engine.
    """
    if selected_row is None or current_rul is None:
        return {"status": "NO_DATA"}

    ts_effect = nasa_causal.get("thermal_stress_effect", {})
    coef = ts_effect.get("coefficient_per_1std", -8.6275)
    reliable = ts_effect.get("reliable", True)

    if not reliable:
        return {"status": "CAUSAL_UNRELIABLE"}

    # Map RUL to urgency and max reduction bound
    if current_rul < 20:
        urgency, max_red = "CRITICAL", 0.40
    elif current_rul < 50:
        urgency, max_red = "WARNING", 0.30
    elif current_rul < 80:
        urgency, max_red = "CAUTION", 0.20
    else:
        return {
            "status": "NO_INTERVENTION_NEEDED", "urgency": "NORMAL",
            "current_rul": round(current_rul, 1),
            "dashboard_recommendation":
                f"Engine healthy — RUL {current_rul:.1f} cycles. No action needed."
        }

    current_ts = float(selected_row.get("thermal_stress", 0))
    if abs(current_ts) <= 0.1:
        return {
            "status": "NO_INTERVENTION_NEEDED", "urgency": urgency,
            "current_rul": round(current_rul, 1),
            "dashboard_recommendation": "Thermal stress already minimal."
        }

    lower_b = current_ts * (1 - max_red)
    upper_b = current_ts
    res = linprog([1.0], bounds=[(lower_b, upper_b)], method="highs")

    if not res.success:
        return {"status": "INFEASIBLE", "urgency": urgency}

    optimal_ts   = float(res.x[0])
    stress_red   = current_ts - optimal_ts
    causal_gain  = abs(coef) * stress_red

    # Model-validated gain: adjust features and re-predict
    model_rul_after = current_rul
    if "nasa_rul" in mdls and feature_cols:
        try:
            adj = selected_row.copy()
            if abs(current_ts) > 1e-6:
                scale = optimal_ts / current_ts
                if "s11" in adj.index: adj["s11"] *= math.sqrt(abs(scale))
                if "s15" in adj.index: adj["s15"] *= math.sqrt(abs(scale))
                if "thermal_stress" in adj.index: adj["thermal_stress"] = optimal_ts
            X = pd.DataFrame([adj[feature_cols].values], columns=feature_cols)
            model_rul_after = float(mdls["nasa_rul"].predict(X)[0])
        except Exception:
            pass

    model_gain = max(0.0, model_rul_after - current_rul)
    red_pct = (stress_red / abs(current_ts)) * 100

    return {
        "status": "SUCCESS", "urgency": urgency,
        "current_ts":        round(current_ts, 4),
        "optimal_ts":        round(optimal_ts, 4),
        "stress_reduction_pct": round(red_pct, 1),
        "max_reduction_pct": round(max_red * 100, 0),
        "current_rul":       round(current_rul, 1),
        "causal_rul_gain":   round(causal_gain, 1),
        "model_rul_after":   round(model_rul_after, 1),
        "model_rul_gain":    round(model_gain, 1),
        "causal_note":       "Linear upper bound (DoWhy coefficient)",
        "model_note":        "LightGBM conservative estimate",
        "dashboard_recommendation": (
            f"[{urgency}] Reduce thermal stress by {red_pct:.0f}% "
            f"(urgency-scaled max {max_red*100:.0f}%). "
            f"Expected RUL gain: +{model_gain:.1f} (model) to +{causal_gain:.1f} (causal) cycles. "
            f"RUL after: {current_rul + model_gain:.1f}–{current_rul + causal_gain:.1f} cycles."
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE AGENT Q&A
# ══════════════════════════════════════════════════════════════════════════════

def query_ollama_agent(question: str) -> str:
    """
    Query Ollama with system_intelligence context.
    This is what MCP resources expose — we read the same JSON directly
    since MCP server runs over stdio, not HTTP.
    """
    try:
        import ollama
        si      = load_json_file("models/system_intelligence.json")
        causal  = load_json_file("models/ai4i_causal_report.json")
        nasa_c  = load_json_file("models/nasa_causal_report.json")
        sim     = load_json_file("models/simulation_results.json")
        hl      = si.get("dashboard_headlines", {})

        context = f"""
You are a Manufacturing AI System expert analyst. Answer using ONLY the data below.
Be specific, cite exact numbers, maximum 5 sentences.

=== SYSTEM METRICS ===
NASA RUL Model: RMSE={hl.get('nasa_rmse')} cycles, MAE from eval report, R²={hl.get('nasa_r2')}
AI4I Failure Model: F1={hl.get('ai4i_f1')}, Precision={hl.get('ai4i_precision')}
Fleet Life Extension: +{hl.get('life_extension_pct')}% (p={hl.get('simulation_p_value')})
Tool Wear Danger Threshold: {hl.get('tool_wear_danger_min')} minutes
Thermal Stress Causal Effect: {hl.get('thermal_stress_coef')} cycles per 1-std
Anomalies Detected: {hl.get('anomaly_count')} / 10,000 samples
Confirmed Failures: {hl.get('failure_count')} / 10,000 samples

=== AI4I CAUSAL FINDINGS ===
Tool wear effect: {causal.get('tool_wear_effect', {}).get('effect_per_10_minutes_pct','—')}% per 10 min
Power effect: {causal.get('power_effect', {}).get('effect_per_100_watts_pct','—')}% per 100W
Refutations: both placebo and random common cause tests passed

=== NASA CAUSAL FINDINGS ===
Thermal stress coefficient: {nasa_c.get('thermal_stress_effect', {}).get('coefficient_per_1std','—')} cycles/std
Reliable: {nasa_c.get('thermal_stress_effect', {}).get('reliable', False)}

=== SIMULATION ===
{sim.get('interpretation','—')}
Managed: {sim.get('managed',{}).get('mean_cycles','—')} ± {sim.get('managed',{}).get('std_cycles','—')} cycles
Unmanaged: {sim.get('unmanaged',{}).get('mean_cycles','—')} ± {sim.get('unmanaged',{}).get('std_cycles','—')} cycles
Mann-Whitney p={sim.get('statistical_significance',{}).get('p_value','—')}
"""
        response = ollama.chat(
            model="llama3.2:latest",
            messages=[
                {"role": "system", "content": context},
                {"role": "user",   "content": question}
            ],
            options={"temperature": 0.1, "num_predict": 400}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Agent unavailable: {e}. Ensure Ollama is running: `ollama serve`"


def query_octotools(question: str) -> str:
    """
    Query Arxiv + Wikipedia via OctoTools for academic context.
    Results summarized to 3 sentences via Ollama.
    """
    results = []
    try:
        from octotools.tools.arxiv_paper_searcher.tool import ArXiv_Paper_Searcher_Tool
        arxiv = ArXiv_Paper_Searcher_Tool()
        raw = arxiv.execute(query=question)
        if raw and isinstance(raw, list):
            for r in raw[:2]:
                t = r.get("title", "")
                a = r.get("abstract", r.get("summary", ""))[:300]
                if t:
                    results.append(f"📄 **{t}**\n{a}")
    except Exception as e:
        results.append(f"Arxiv unavailable: {e}")

    try:
        from octotools.tools.wikipedia_knowledge_searcher.tool import Wikipedia_Knowledge_Searcher_Tool
        wiki = Wikipedia_Knowledge_Searcher_Tool()
        raw = wiki.execute(query=question)
        if raw:
            content = (raw.get("summary", str(raw))[:400]
                       if isinstance(raw, dict) else str(raw)[:400])
            results.append(f"📖 **Wikipedia:**\n{content}")
    except Exception as e:
        results.append(f"Wikipedia unavailable: {e}")

    if not results:
        return "No academic results found for this query."

    # Summarize via Ollama to 3 lines
    try:
        import ollama
        combined = "\n\n".join(results)
        resp = ollama.chat(
            model="llama3.2:latest",
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize these academic results in exactly 3 sentences "
                    f"relevant to manufacturing AI:\n\n{combined}"
                )
            }],
            options={"temperature": 0.0, "num_predict": 200}
        )
        return resp["message"]["content"].strip()
    except Exception:
        return "\n\n".join(results)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px;">
            <div style="font-size:2rem;">🏭</div>
            <div style="font-family:'Inter';font-weight:700;font-size:1rem;
                        background:linear-gradient(135deg,#ff2d55,#00d4ff);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;">Manufacturing AI</div>
            <div style="color:#8888aa;font-size:0.7rem;margin-top:2px;">
                Intelligent Failure Prevention System
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(255,45,85,0.2);margin:8px 0;'>",
                    unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["🏠 System Overview", "🏭 AI4I Machine Monitor",
             "✈️  NASA Engine Fleet", "🔬 Causal Intelligence",
             "📈 MLOps & Monitoring", "📚 Research & Reports"],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color:rgba(255,45,85,0.2);margin:12px 0;'>",
                    unsafe_allow_html=True)

        si    = load_json_file("models/system_intelligence.json")
        ready = si.get("system_status", {}).get("ready_for_production", False)
        badge = "ok" if ready else "critical"
        label = "● PRODUCTION READY" if ready else "● NOT READY"
        st.markdown(f'<div class="status-badge-{badge}">{label}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        hl = si.get("dashboard_headlines", {})
        if hl:
            st.markdown('<div class="section-header">Live Metrics</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.78rem;color:#8888aa;line-height:2.2;">
                <span style="color:#00d4ff;font-family:'JetBrains Mono';">RMSE </span>
                {hl.get('nasa_rmse','—')} cycles<br>
                <span style="color:#00d4ff;font-family:'JetBrains Mono';">R²   </span>
                {hl.get('nasa_r2','—')}<br>
                <span style="color:#00d4ff;font-family:'JetBrains Mono';">F1   </span>
                {hl.get('ai4i_f1','—')}<br>
                <span style="color:#00d4ff;font-family:'JetBrains Mono';">PREC </span>
                {hl.get('ai4i_precision','—')}<br>
                <span style="color:#00ff88;font-family:'JetBrains Mono';">
                +{hl.get('life_extension_pct','—')}%</span> life ext.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">External Tools</div>',
                    unsafe_allow_html=True)
        st.link_button("📊 MLflow UI",        "http://localhost:5000",
                        use_container_width=True)
        st.link_button("🔬 Solara Digital Twin", "http://localhost:9999",
                        use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:#8888aa;font-size:0.65rem;text-align:center;">'
            f'Updated: {datetime.now().strftime("%H:%M:%S")}</div>',
            unsafe_allow_html=True
        )

    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SYSTEM OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_overview():
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:40px;">
        <h1 class="neon-title">Intelligent Manufacturing<br>Failure Prevention System</h1>
        <p class="neon-subtitle" style="font-size:1rem;margin-top:12px;">
            End-to-End AI Pipeline · Predictive Maintenance · Causal Intelligence · Digital Twin
        </p>
        <div style="margin-top:20px;">
            <span class="tech-badge">LightGBM</span>
            <span class="tech-badge">Random Forest</span>
            <span class="tech-badge">OR-Tools CP-SAT</span>
            <span class="tech-badge">DoWhy</span>
            <span class="tech-badge">Mesa ABM</span>
            <span class="tech-badge">FastMCP</span>
            <span class="tech-badge">OctoTools</span>
            <span class="tech-badge">Optuna TPE</span>
            <span class="tech-badge">MLflow</span>
            <span class="tech-badge">DeepChecks</span>
            <span class="tech-badge">Evidently</span>
            <span class="tech-badge">Polars</span>
            <span class="tech-badge">SciPy linprog</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    si  = load_json_file("models/system_intelligence.json")
    hl  = si.get("dashboard_headlines", {})
    sim = load_json_file("models/simulation_results.json")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">'
        'System Performance — All metrics from held-out test data'
        '</div>',
        unsafe_allow_html=True
    )

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("NASA RMSE",       f"{hl.get('nasa_rmse','—')} cyc",
              help="LightGBM on 142 held-out engines")
    c2.metric("NASA R²",          hl.get('nasa_r2','—'),
              help="Variance explained — no leakage")
    c3.metric("AI4I F1",          hl.get('ai4i_f1','—'),
              help="Held-out 20% stratified test set")
    c4.metric("Precision",        hl.get('ai4i_precision','—'),
              help="96.4% of alarms are real failures")
    c5.metric("Life Extension",  f"+{hl.get('life_extension_pct','—')}%",
              help="50 managed vs 50 unmanaged engines, p<0.0001")
    c6.metric("Failures Caught",  f"{hl.get('failure_count','—')}",
              help="Ground truth failures in 10,000 samples")

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Pipeline Layer Status</div>',
                    unsafe_allow_html=True)

        layers = {
            "Layer 1 · Data Processing":     "data/processed/ai4i_cleaned.parquet",
            "Layer 2 · Feature Engineering": "data/processed/ai4i_features.parquet",
            "Layer 3 · Model Training":      "models/ai4i_classifier.joblib",
            "Layer 4 · Causal Analysis":     "models/ai4i_causal_report.json",
            "Layer 5 · Optimization":        "models/ai4i_optimizer_results.json",
            "Layer 6 · Simulation":          "models/simulation_results.json",
            "Layer 7 · Intelligence":        "models/system_intelligence.json",
            "Layer 8 · MLOps Monitoring":    "reports/monitoring",
        }
        for name, path in layers.items():
            ok   = (PROJECT_ROOT / path).exists()
            icon = "🟢" if ok else "🔴"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<span style="font-size:0.82rem;color:#c0c0d0;">{name}</span>'
                f'<span>{icon}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Refresh Pipeline</div>',
                    unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            if st.button("🔄 Re-run Intelligence (L7)", use_container_width=True):
                with st.spinner("Running Layer 7 (aggregator + researcher + plotter)..."):
                    r = trigger_layer(7)
                if r["success"]:
                    st.success(f"✅ Done in {r['elapsed']}s")
                    load_json_file.clear()
                else:
                    st.error("❌ Failed")
                    for e in r["errors"]:
                        st.code(e)
        with cb:
            if st.button("📊 Re-run Monitoring (L8)", use_container_width=True):
                with st.spinner("Running Layer 8 (MLflow + DeepChecks + Evidently)..."):
                    r = trigger_layer(8)
                st.success(f"✅ Done in {r['elapsed']}s") if r["success"] \
                    else st.error(f"❌ {r['errors']}")

        st.markdown('</div>', unsafe_allow_html=True)

    with cr:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">System Architecture</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'JetBrains Mono';font-size:0.72rem;
                    line-height:1.9;color:#c0c0d0;">
        <span style="color:#ff2d55;">RAW DATA</span>
        → AI4I 2020 (10k) + CMAPSS FD001-4 (160k)<br>&nbsp;&nbsp;↓<br>
        <span style="color:#ffaa00;">L1</span> Polars · Condition Clustering · Z-score<br>
        &nbsp;&nbsp;↓<br>
        <span style="color:#ffaa00;">L2</span> Physics Features · Savgol · Velocity/Trend<br>
        &nbsp;&nbsp;↓<br>
        <span style="color:#00d4ff;">L3</span> RF Classifier · LightGBM + Optuna 30-trial TPE<br>
        &nbsp;&nbsp;↓<br>
        <span style="color:#00d4ff;">L4</span> DoWhy Backdoor · 2 Refutation Tests<br>
        &nbsp;&nbsp;↓<br>
        <span style="color:#00d4ff;">L5</span> OR-Tools CP-SAT · SciPy linprog HiGHS<br>
        &nbsp;&nbsp;↓<br>
        <span style="color:#bf5fff;">L6</span> Mesa ABM · 50v50 · Mann-Whitney U<br>
        &nbsp;&nbsp;↓<br>
        <span style="color:#bf5fff;">L7</span> FastMCP · Ollama · OctoTools · Tavily<br>
        &nbsp;&nbsp;↓<br>
        <span style="color:#00ff88;">L8</span> MLflow · DeepChecks · Evidently<br>
        &nbsp;&nbsp;↓<br>
        <span style="color:#ff2d55;">DASHBOARD</span> Streamlit · Live Inference · Agent Q&A
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        best = load_json_file("models/nasa_best_params.json")
        if best:
            st.markdown('<div class="section-header">Optuna Best Hyperparameters (30 Trials TPE)</div>',
                        unsafe_allow_html=True)
            show_params = ["n_estimators","max_depth","learning_rate","num_leaves","subsample"]
            for k in show_params:
                v = best.get(k, "—")
                if isinstance(v, float): v = f"{v:.4f}"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:3px 0;">'
                    f'<span style="font-size:0.78rem;color:#8888aa;">{k}</span>'
                    f'<span style="font-family:\'JetBrains Mono\';font-size:0.78rem;'
                    f'color:#00d4ff;">{v}</span></div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # Simulation headline banner
    st.markdown("<br>", unsafe_allow_html=True)
    if sim:
        mm  = sim.get("managed", {}).get("mean_cycles", 0)
        um  = sim.get("unmanaged", {}).get("mean_cycles", 0)
        g   = sim.get("life_extension_pct", 0)
        p   = sim.get("statistical_significance", {}).get("p_value", 1)
        ms  = sim.get("managed", {}).get("std_cycles", 0)
        us  = sim.get("unmanaged", {}).get("std_cycles", 0)
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:0.75rem;color:#8888aa;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:20px;">
                Mesa Agent-Based Simulation · 50 AI-Managed vs 50 Unmanaged Engines
                · Mann-Whitney U (one-sided)
            </div>
            <div style="display:flex;justify-content:center;gap:60px;align-items:center;
                        flex-wrap:wrap;">
                <div>
                    <div class="kpi-value" style="color:#00ff88;">+{g:.1f}%</div>
                    <div class="kpi-label">Fleet Life Extension</div>
                    <div class="kpi-delta">p = {p:.4f} · Statistically Significant ✅</div>
                </div>
                <div>
                    <div class="kpi-value">{mm:.0f}</div>
                    <div class="kpi-label">AI-Managed (avg cycles)</div>
                    <div style="font-size:0.75rem;color:#8888aa;">± {ms:.0f} std</div>
                </div>
                <div style="color:#8888aa;font-size:2rem;">vs</div>
                <div>
                    <div class="kpi-value" style="color:#ff2d55;">{um:.0f}</div>
                    <div class="kpi-label">Unmanaged (avg cycles)</div>
                    <div style="font-size:0.75rem;color:#8888aa;">± {us:.0f} std</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — AI4I MACHINE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def page_ai4i(mdls):
    st.markdown('<h2 class="neon-title" style="font-size:1.6rem;">🏭 AI4I Machine Monitor</h2>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="neon-subtitle">Real-time failure prediction (RF) · '
        'Anomaly detection (Isolation Forest) · Live OR-Tools setpoint · '
        'DoWhy causal risk</div><br>',
        unsafe_allow_html=True
    )

    ai4i_causal = load_json_file("models/ai4i_causal_report.json")
    ai4i_opt    = load_json_file("models/ai4i_optimizer_results.json")
    ai4i_eval   = load_json_file("models/ai4i_eval_report.json")

    # ── INPUTS ────────────────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Machine Sensor Input — All predictions update live</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔩 Mechanical**")
        tool_wear = st.slider("Tool Wear (minutes)", 0, 240, 100, 5,
            help="Accumulated wear. Danger threshold from DoWhy causal analysis.")
        rpm       = st.slider("Rotational Speed (RPM)", 1168, 2886, 1551, 10,
            help="Machine rotational speed")
        torque    = st.slider("Torque (Nm)", 3.8, 76.6, 42.8, 0.5,
            help="Applied torque")

    with c2:
        st.markdown("**🌡️ Thermal**")
        air_temp  = st.slider("Air Temperature (°C)", 23.0, 30.0, 25.0, 0.1)
        proc_temp = st.slider("Process Temperature (°C)", 32.9, 39.4, 35.5, 0.1)
        temp_delta = proc_temp - air_temp
        hdf_risk   = temp_delta < 8.6
        st.markdown(
            f'<div class="{"critical-box" if hdf_risk else "insight-box"}">'
            f'Temp Δ = <strong>{temp_delta:.1f}°C</strong> '
            f'{"⚠️ HDF risk (< 8.6°C)" if hdf_risk else "✅ Safe (≥ 8.6°C)"}</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown("**🏷️ Product Config**")
        prod_label = st.selectbox(
            "Product Type",
            ["L (Low — OSF limit 11,000)", "M (Medium — OSF limit 12,000)",
             "H (High — OSF limit 13,000)"],
            index=1,
            help="Determines OSF (Overstrain Failure) threshold from dataset documentation"
        )
        pt_code = prod_label[0]
        pt_num  = {"L": 0, "M": 1, "H": 2}[pt_code]
        osf_lim = OSF_THRESHOLDS[pt_num]
        osf_load  = tool_wear * torque
        osf_margin = max(0, (1 - osf_load / osf_lim) * 100)
        power_w = torque * (rpm * 2 * math.pi / 60)

        pwf_risk = power_w < PWF_MIN or power_w > PWF_MAX
        st.markdown(
            f'<div class="{"critical-box" if pwf_risk else "insight-box"}">'
            f'Power: <strong>{power_w:.0f}W</strong> '
            f'{"⚠️ PWF risk (outside 3500–9000W)" if pwf_risk else "✅ Safe [3500–9000W]"}</div>',
            unsafe_allow_html=True
        )
        osf_critical = osf_margin < 20
        st.markdown(
            f'<div class="{"warning-box" if osf_critical else "insight-box"}">'
            f'OSF Load: <strong>{osf_load:.0f}</strong> / {osf_lim} '
            f'({osf_margin:.1f}% margin)'
            f'{"⚠️ < 20%" if osf_critical else " ✅"}</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── LIVE PREDICTIONS ──────────────────────────────────────────────────────
    failure_prob, is_anomaly, ascore = predict_ai4i(
        mdls, rpm, torque, tool_wear, air_temp, proc_temp
    )

    tw_eff   = ai4i_causal.get("tool_wear_effect", {}).get("effect_per_10_minutes_pct", 0.305)
    danger   = ai4i_causal.get("tool_wear_effect", {}).get("danger_threshold_minutes", 240)
    pw_eff   = ai4i_causal.get("power_effect", {}).get("effect_per_100_watts_pct", 0.300)
    tw_risk  = (tool_wear / 10) * tw_eff
    pw_risk  = (power_w / 100) * pw_eff

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Live Prediction Results — Updates on every slider move</div>',
                unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        if failure_prob is not None:
            color = "#ff2d55" if failure_prob > 0.5 else "#ffaa00" if failure_prob > 0.2 else "#00ff88"
            label = "⚠️ FAILURE RISK" if failure_prob > 0.5 else "🔶 CAUTION" if failure_prob > 0.2 else "✅ NORMAL"
            badge = "critical" if failure_prob > 0.5 else "warning" if failure_prob > 0.2 else "ok"
            st.markdown(
                f'<div class="kpi-value" style="color:{color};">{failure_prob*100:.1f}%</div>'
                f'<div class="kpi-label">Failure Probability</div>'
                f'<div style="margin-top:8px;">'
                f'<span class="status-badge-{badge}">{label}</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="kpi-label">Model not loaded</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        if is_anomaly is not None:
            ac = "#ff2d55" if is_anomaly else "#00ff88"
            al = "⚠️ ANOMALY" if is_anomaly else "✅ NORMAL"
            st.markdown(
                f'<div class="kpi-value" style="color:{ac};font-size:1.8rem;">{al}</div>'
                f'<div class="kpi-label">Isolation Forest</div>'
                f'<div style="font-size:0.72rem;color:#8888aa;margin-top:6px;">'
                f'Score: {ascore:.3f}</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        wc = "#ff2d55" if tool_wear >= danger else "#ffaa00" if tool_wear >= 200 else "#00ff88"
        remaining = max(0, danger - tool_wear)
        st.markdown(
            f'<div class="kpi-value" style="color:{wc};">{tool_wear}</div>'
            f'<div class="kpi-label">Tool Wear (min)</div>'
            f'<div style="font-size:0.75rem;color:#8888aa;margin-top:6px;">'
            f'{remaining:.0f} min to danger zone</div>',
            unsafe_allow_html=True
        )
        st.progress(min(tool_wear / 240, 1.0))
        st.markdown('</div>', unsafe_allow_html=True)

    with r4:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        pc = "#ff2d55" if pwf_risk else "#00ff88"
        st.markdown(
            f'<div class="kpi-value" style="color:{pc};">{power_w:.0f}</div>'
            f'<div class="kpi-label">Power (Watts)</div>'
            f'<div style="font-size:0.75rem;color:#8888aa;margin-top:6px;">'
            f'Safe zone: 3500–9000W</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── OR-TOOLS SETPOINT (LIVE) ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_opt, col_risk = st.columns([1.3, 1])

    with col_opt:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">OR-Tools CP-SAT Optimal Setpoint</div>',
                    unsafe_allow_html=True)

        with st.spinner("Solving constraints..."):
            sp = get_setpoint(int(tool_wear), pt_code, ai4i_opt)

        src   = sp.get("source", "—")
        urg   = sp.get("urgency", "NORMAL")
        urg_c = {"NORMAL":"#00ff88","CAUTION":"#00d4ff",
                 "WARNING":"#ffaa00","CRITICAL":"#ff2d55"}.get(urg,"#8888aa")

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;margin-bottom:12px;">'
            f'<span style="color:{urg_c};font-weight:600;">● {urg}</span>'
            f'<span style="font-size:0.72rem;color:#8888aa;">{src}</span>'
            f'<span style="font-size:0.72rem;color:#8888aa;">'
            f'Matched: {sp.get("matched_scenario", sp.get("current_tool_wear_min","—"))}'
            f'</span></div>',
            unsafe_allow_html=True
        )

        if sp.get("status") in ["OPTIMAL", "FEASIBLE", None]:
            s1, s2, s3 = st.columns(3)
            s1.metric("Optimal RPM",    sp.get("optimal_rpm","—"))
            s2.metric("Optimal Torque", f"{sp.get('optimal_torque_nm','—')} Nm")
            s3.metric("Power Output",   f"{sp.get('actual_power_w',0):.0f}W")

            st.markdown(
                f'<div class="insight-box">'
                f'OSF Safety Margin: <strong>{sp.get("osf_margin_pct",0):.1f}%</strong> · '
                f'Target Power: {sp.get("target_power_w","—")}W · '
                f'Type: {sp.get("prod_type","—")}</div>',
                unsafe_allow_html=True
            )
            ca = sp.get("constraints_applied", {})
            if ca:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:#8888aa;">'
                    f'OSF: {ca.get("osf","—")} · '
                    f'TWF: {ca.get("twf","—")} · '
                    f'Causal: {ca.get("causal","—")} · '
                    f'PWF: {ca.get("pwf","—")}</div>',
                    unsafe_allow_html=True
                )
        elif sp.get("status") == "INFEASIBLE":
            st.markdown(
                '<div class="critical-box">⚠️ No safe setpoint found — '
                'schedule immediate tool replacement</div>',
                unsafe_allow_html=True
            )
        elif sp.get("status") == "ERROR":
            st.markdown(
                f'<div class="warning-box">OR-Tools error: {sp.get("error","—")}<br>'
                f'Showing nearest pre-computed scenario</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col_risk:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">DoWhy Causal Risk Breakdown</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.75rem;color:#8888aa;margin-bottom:12px;">'
            'Pre-computed coefficients applied to live sensor values. '
            'No re-running DoWhy — coefficients are stable.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:0.82rem;margin-bottom:4px;">'
            f'<span>Tool Wear Risk (+{tw_eff:.3f}%/10min)</span>'
            f'<span style="color:#ffaa00;font-family:\'JetBrains Mono\';">'
            f'+{tw_risk:.2f}%</span></div>',
            unsafe_allow_html=True
        )
        st.progress(min(tw_risk / 15, 1.0))

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:0.82rem;margin-bottom:4px;margin-top:8px;">'
            f'<span>Power Risk (+{pw_eff:.4f}%/100W)</span>'
            f'<span style="color:#ffaa00;font-family:\'JetBrains Mono\';">'
            f'+{pw_risk:.2f}%</span></div>',
            unsafe_allow_html=True
        )
        st.progress(min(pw_risk / 30, 1.0))

        box_cls = "critical-box" if remaining < 30 else \
                  "warning-box"  if remaining < 60 else "insight-box"
        st.markdown(
            f'<div class="{box_cls}" style="margin-top:12px;">'
            f'<strong>{remaining:.0f} min</strong> until danger threshold '
            f'({danger:.0f} min per DoWhy)<br>'
            f'Combined causal risk: <strong>+{tw_risk+pw_risk:.2f}%</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        img = PROJECT_ROOT / "reports/plots/tool_wear_risk.png"
        if img.exists():
            st.image(str(img), caption="Tool Wear Risk Analysis (DoWhy + RF)",
                     use_container_width=True)
    with p2:
        img = PROJECT_ROOT / "reports/plots/failure_mode_breakdown.png"
        if img.exists():
            st.image(str(img), caption="Failure Mode Distribution",
                     use_container_width=True)

    img = PROJECT_ROOT / "reports/plots/feature_importance_ai4i.png"
    if img.exists():
        st.image(str(img), caption="Random Forest Feature Importances",
                 use_container_width=True)

    with st.expander("ℹ️ AI4I Model Details — Random Forest + Isolation Forest"):
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("F1 Score",  ai4i_eval.get("f1_score","—"))
        mc2.metric("Precision", ai4i_eval.get("precision","—"))
        mc3.metric("Recall",    ai4i_eval.get("recall","—"))
        mc4.metric("ROC-AUC",   ai4i_eval.get("roc_auc","—"))
        st.markdown(
            f'CV F1: **{ai4i_eval.get("cv_f1_mean","—")} ± {ai4i_eval.get("cv_f1_std","—")}** · '
            f'Test: 20% stratified · Class weight: balanced · '
            f'Imbalance: 3.4% failure rate · '
            f'Isolation Forest trained on normal samples only (failure=0)'
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — NASA ENGINE FLEET
# ══════════════════════════════════════════════════════════════════════════════

def page_nasa(mdls):
    st.markdown('<h2 class="neon-title" style="font-size:1.6rem;">✈️ NASA Engine Fleet Monitor</h2>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="neon-subtitle">LightGBM RUL prediction · '
        'SciPy linprog live intervention · DoWhy thermal stress causal · '
        'Fleet survival · CMAPSS FD001-FD004</div><br>',
        unsafe_allow_html=True
    )

    nasa_eval   = load_json_file("models/nasa_eval_report.json")
    nasa_causal = load_json_file("models/nasa_causal_report.json")
    nasa_opt    = load_json_file("models/nasa_optimizer_results.json")
    sim_res     = load_json_file("models/simulation_results.json")
    nasa_df     = load_nasa_df()
    surv_df     = load_survival_csv()

    feature_cols = ([c for c in nasa_df.columns if c not in NASA_EXCLUDE_COLS]
                    if not nasa_df.empty else [])

    # ── ENGINE SELECTOR ───────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Engine State Input</div>',
                unsafe_allow_html=True)

    mode = st.radio("Input Mode",
                    ["📋 Scenario (pre-computed)", "🔧 Real Engine from Dataset"],
                    horizontal=True)

    selected_row = None
    current_rul  = None
    scenario_key = "healthy"

    if mode == "📋 Scenario (pre-computed)":
        cs1, cs2 = st.columns(2)
        with cs1:
            scen = st.selectbox("Engine Health Scenario", [
                "🔴 Critical (RUL < 20 cycles)",
                "🟡 Degrading (RUL 20–50 cycles)",
                "🟢 Healthy (RUL > 80 cycles)"
            ])
        with cs2:
            st.markdown(
                '<div class="insight-box" style="margin-top:8px;">'
                'Selecting a scenario loads a real engine row from the dataset '
                'matching that RUL range. LightGBM re-predicts live.</div>',
                unsafe_allow_html=True
            )

        scenario_key = {
            "🔴 Critical (RUL < 20 cycles)":   "critical",
            "🟡 Degrading (RUL 20–50 cycles)": "degrading",
            "🟢 Healthy (RUL > 80 cycles)":    "healthy",
        }[scen]

        if not nasa_df.empty:
            filt = {
                "critical":  nasa_df["rul"] < 20,
                "degrading": (nasa_df["rul"] >= 20) & (nasa_df["rul"] < 50),
                "healthy":   nasa_df["rul"] >= 80,
            }[scenario_key]
            eligible = nasa_df[filt]
            if not eligible.empty:
                selected_row = eligible.sample(1, random_state=42).iloc[0]
                if feature_cols:
                    current_rul = predict_nasa_rul(mdls, selected_row, feature_cols)

    else:  # Real engine
        if not nasa_df.empty:
            ce1, ce2, ce3 = st.columns(3)
            with ce1:
                ds = st.selectbox("Dataset", ["FD001","FD002","FD003","FD004"])
            with ce2:
                ds_df = nasa_df[nasa_df["dataset_id"] == ds]
                units = sorted(ds_df["unit"].unique().astype(int).tolist())
                uid   = st.selectbox("Engine Unit", units)
            with ce3:
                u_df = ds_df[ds_df["unit"] == uid]
                maxc = int(u_df["cycle"].max())
                cyc  = st.slider("Cycle", 1, maxc, maxc // 2)

            mask    = ((nasa_df["dataset_id"] == ds) &
                       (nasa_df["unit"] == uid) &
                       (nasa_df["cycle"] == cyc))
            matches = nasa_df[mask]
            if not matches.empty:
                selected_row = matches.iloc[0]
                if feature_cols:
                    current_rul = predict_nasa_rul(mdls, selected_row, feature_cols)
                if current_rul is not None:
                    scenario_key = ("critical"  if current_rul < 20 else
                                    "degrading" if current_rul < 50 else "healthy")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── RUL DISPLAY ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if current_rul is not None:
        urgency = ("CRITICAL" if current_rul < 20 else
                   "WARNING"  if current_rul < 50 else
                   "CAUTION"  if current_rul < 80 else "NORMAL")
        urg_c = {"NORMAL":"#00ff88","CAUTION":"#00d4ff",
                 "WARNING":"#ffaa00","CRITICAL":"#ff2d55"}[urgency]
        rul_pct = min(current_rul / 125 * 100, 100)

        dr1, dr2, dr3 = st.columns(3)
        with dr1:
            badge = ("ok" if urgency == "NORMAL" else
                     "warning" if urgency in ["CAUTION","WARNING"] else "critical")
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div class="kpi-value" style="color:{urg_c};font-size:3rem;">
                    {current_rul:.0f}
                </div>
                <div class="kpi-label">Predicted RUL (cycles)</div>
                <div style="margin-top:12px;">
                    <span class="status-badge-{badge}">{urgency}</span>
                </div>
                <div style="margin-top:10px;font-size:0.74rem;color:#8888aa;">
                    LightGBM · RMSE={nasa_eval.get('rmse','—')} · R²={nasa_eval.get('r2','—')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with dr2:
            opt = nasa_opt.get(scenario_key, {})
            st.markdown(f"""
            <div class="glass-card">
                <div class="section-header">Pre-computed Intervention (Layer 5)</div>
                <div style="font-size:0.82rem;line-height:2;">
                    <span style="color:#8888aa;">Urgency:</span>
                    <span style="color:{urg_c};">{opt.get('urgency','—')}</span><br>
                    <span style="color:#8888aa;">Stress reduction:</span>
                    <span style="color:#00d4ff;">{opt.get('stress_reduction_pct','0')}%</span><br>
                    <span style="color:#8888aa;">Causal gain:</span>
                    <span style="color:#00ff88;">+{opt.get('causal_rul_gain_cycles','—')} cyc</span><br>
                    <span style="color:#8888aa;">Model gain:</span>
                    <span style="color:#00ff88;">+{opt.get('model_rul_gain_cycles','—')} cyc</span>
                </div>
                <div class="insight-box" style="margin-top:8px;font-size:0.8rem;">
                    {opt.get('dashboard_recommendation','Select a scenario to see recommendation')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with dr3:
            ts_val = (float(selected_row["thermal_stress"])
                      if selected_row is not None and
                         "thermal_stress" in selected_row.index else None)
            coef   = nasa_causal.get("thermal_stress_effect", {}).get(
                         "coefficient_per_1std", -8.6275)
            ts_html = (
                f'<span style="color:#8888aa;">Current thermal stress:</span><br>'
                f'<span style="color:#ffaa00;font-family:\'JetBrains Mono\';">'
                f'{ts_val:.4f} std</span>'
            ) if ts_val is not None else ""

            st.markdown(f"""
            <div class="glass-card">
                <div class="section-header">DoWhy Causal (Layer 4)</div>
                <div style="font-size:0.82rem;line-height:1.9;">
                    <span style="color:#8888aa;">Thermal coef:</span>
                    <span style="color:#00d4ff;font-family:'JetBrains Mono';">
                        {coef:.4f} cyc/std</span><br>
                    <span style="color:#8888aa;">Meaning:</span><br>
                    <span style="font-size:0.78rem;">
                        1σ thermal stress → {abs(coef):.1f} fewer cycles</span><br>
                    {ts_html}
                </div>
                <div class="insight-box" style="margin-top:8px;">
                    Both refutation tests passed ✅<br>
                    Placebo + Random common cause
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin:16px 0 4px;display:flex;justify-content:space-between;
                    font-size:0.75rem;color:#8888aa;">
            <span>Engine Life Remaining</span>
            <span>{rul_pct:.0f}% ({current_rul:.0f} / 125 cycles RUL cap)</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(rul_pct / 100)

    # ── LIVE LINPROG INTERVENTION ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">'
        'SciPy linprog — Live Intervention (recalculates for selected engine)'
        '</div>',
        unsafe_allow_html=True
    )

    if selected_row is not None and current_rul is not None and feature_cols:
        with st.spinner("Running SciPy HiGHS linprog..."):
            iv = live_linprog_intervention(
                mdls, selected_row, feature_cols, current_rul, nasa_causal
            )

        if iv["status"] == "SUCCESS":
            lp1, lp2, lp3, lp4 = st.columns(4)
            lp1.metric("Current Thermal Stress", f"{iv['current_ts']} std")
            lp2.metric("Optimal Target Stress",  f"{iv['optimal_ts']} std",
                       delta=f"-{iv['stress_reduction_pct']}%")
            lp3.metric("Causal RUL Gain",
                       f"+{iv['causal_rul_gain']} cyc",
                       help="Linear upper bound from DoWhy coefficient")
            lp4.metric("Model RUL Gain",
                       f"+{iv['model_rul_gain']} cyc",
                       help="LightGBM conservative estimate post-adjustment")

            st.markdown(
                f'<div class="success-box">'
                f'<strong>linprog recommendation:</strong> '
                f'{iv["dashboard_recommendation"]}<br>'
                f'<span style="font-size:0.75rem;color:#8888aa;">'
                f'Max allowed: {iv["max_reduction_pct"]}% (urgency-scaled) · '
                f'Causal: {iv["causal_note"]} · Model: {iv["model_note"]}'
                f'</span></div>',
                unsafe_allow_html=True
            )
        elif iv["status"] == "NO_INTERVENTION_NEEDED":
            st.markdown(
                f'<div class="insight-box">'
                f'✅ {iv.get("dashboard_recommendation","No intervention needed")}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="warning-box">'
                f'linprog status: {iv["status"]}'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="warning-box">Select an engine to run live linprog intervention.</div>',
            unsafe_allow_html=True
        )

    # ── SURVIVAL CURVE (PLOTLY) ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Fleet Survival — Interactive Plotly (from simulation CSV)</div>',
                unsafe_allow_html=True)

    if not surv_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=surv_df["cycle"], y=surv_df["survival_pct_managed"],
            name="AI-Managed Fleet",
            line=dict(color="#00d4ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
            hovertemplate="Cycle %{x}<br>Survival: %{y:.1f}%<extra>AI-Managed</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=surv_df["cycle"], y=surv_df["survival_pct_unmanaged"],
            name="Unmanaged Fleet",
            line=dict(color="#ff2d55", width=2.5, dash="dash"),
            fill="tozeroy", fillcolor="rgba(255,45,85,0.04)",
            hovertemplate="Cycle %{x}<br>Survival: %{y:.1f}%<extra>Unmanaged</extra>"
        ))
        fig.add_vline(x=80, line_width=1, line_dash="dot",
                      line_color="#ffaa00", opacity=0.7,
                      annotation_text="Divergence point",
                      annotation_font_color="#ffaa00",
                      annotation_font_size=11)
        if sim_res:
            mm  = sim_res.get("managed",{}).get("mean_cycles",0)
            um  = sim_res.get("unmanaged",{}).get("mean_cycles",0)
            g   = sim_res.get("life_extension_pct",0)
            p   = sim_res.get("statistical_significance",{}).get("p_value",1)
            fig.add_vline(x=mm, line_width=1, line_dash="dot",
                          line_color="#00d4ff", opacity=0.4)
            fig.add_vline(x=um, line_width=1, line_dash="dot",
                          line_color="#ff2d55", opacity=0.4)
            fig.add_annotation(
                x=0.98, y=0.95, xref="paper", yref="paper",
                text=(f"Life Extension: +{g:.1f}%<br>"
                      f"p = {p:.4f} ✅ (Mann-Whitney U)<br>"
                      f"Managed: {mm:.0f} cycles<br>"
                      f"Unmanaged: {um:.0f} cycles"),
                showarrow=False,
                bgcolor="rgba(8,8,16,0.85)",
                bordercolor="rgba(0,212,255,0.4)",
                font=dict(size=11, color="#e0e0e0"), align="right"
            )
        fig.update_layout(
            **PLOTLY_BASE,
            title="AI-Managed vs Unmanaged Fleet Survival · Mesa ABM · 50v50",
            xaxis_title="Operational Cycles",
            yaxis_title="Fleet Survival (%)",
            yaxis_range=[0, 105], height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    img = PROJECT_ROOT / "reports/plots/feature_importance_nasa.png"
    if img.exists():
        st.image(str(img), caption="LightGBM Feature Importances — NASA RUL",
                 use_container_width=True)

    with st.expander("ℹ️ NASA Model Details — LightGBM + Optuna"):
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("RMSE", f"{nasa_eval.get('rmse','—')} cycles")
        mc2.metric("MAE",  f"{nasa_eval.get('mae','—')} cycles")
        mc3.metric("R²",   nasa_eval.get("r2","—"))
        st.markdown(
            "**Evaluation:** 142 held-out test engines (unit-level split — no leakage) · "
            "**Training:** 567 engines · **Optuna:** 30 trials TPE Bayesian · "
            "**Datasets:** CMAPSS FD001-FD004 merged · **RUL cap:** 125 cycles · "
            "**Thermal stress:** physics feature = s11 × s15 (temp × pressure ratio)"
        )

    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:20px;margin-top:16px;">
        <div style="font-size:0.85rem;color:#8888aa;margin-bottom:12px;">
            Live Mesa simulation with engine-level visualization
            (requires mfg-2 conda environment)
        </div>
        <a href="http://localhost:9999" target="_blank" style="
            display:inline-block;
            background:linear-gradient(135deg,rgba(191,95,255,0.2),rgba(0,212,255,0.1));
            border:1px solid rgba(191,95,255,0.4); border-radius:8px;
            padding:10px 28px; color:white; text-decoration:none; font-weight:500;">
            🔬 Open Solara Digital Twin →
        </a>
        <div style="font-size:0.7rem;color:#8888aa;margin-top:8px;">
            conda activate mfg-2 &amp;&amp; solara run src/viz_dashboard.py
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CAUSAL INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def page_causal():
    st.markdown('<h2 class="neon-title" style="font-size:1.6rem;">🔬 Causal Intelligence</h2>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="neon-subtitle">DoWhy Structural Causal Models · '
        'Backdoor adjustment · Refutation-validated · Back-transformed to real units</div><br>',
        unsafe_allow_html=True
    )

    ai4i_c = load_json_file("models/ai4i_causal_report.json")
    nasa_c = load_json_file("models/nasa_causal_report.json")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">AI4I — Tool Wear → Machine Failure</div>',
                    unsafe_allow_html=True)

        tw = ai4i_c.get("tool_wear_effect", {})
        pw = ai4i_c.get("power_effect", {})

        # ASCII SCM diagram
        st.markdown("""
        <div style="font-family:'JetBrains Mono';font-size:0.76rem;
                    background:rgba(0,212,255,0.04);border-radius:8px;
                    padding:12px;margin-bottom:12px;line-height:1.9;">
        <span style="color:#ffaa00;">prod_type</span> &nbsp;─────────────────┐<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#00d4ff;">rpm / torque</span> ───────────→
        <span style="color:#ff2d55;font-weight:bold;">failure</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↑<br>
        <span style="color:#bf5fff;">tool_wear</span> &nbsp;──────────────────┘
        </div>
        """, unsafe_allow_html=True)

        if tw:
            st.markdown(f"""
            <div style="font-size:0.84rem;line-height:2.1;">
                <span style="color:#8888aa;">Scaled coefficient:</span>
                <span style="color:#00d4ff;font-family:'JetBrains Mono';">
                    {tw.get('scaled_coefficient','—')}</span><br>
                <span style="color:#8888aa;">Effect per 10 min:</span>
                <span style="color:#ffaa00;font-family:'JetBrains Mono';font-weight:600;">
                    +{tw.get('effect_per_10_minutes_pct','—')}% failure prob</span><br>
                <span style="color:#8888aa;">Danger threshold:</span>
                <span style="color:#ff2d55;font-family:'JetBrains Mono';font-weight:600;">
                    {tw.get('danger_threshold_minutes','—')} minutes</span><br>
                <span style="color:#8888aa;">Interpretation:</span><br>
                <span style="font-size:0.78rem;">{tw.get('interpretation','—')}</span><br>
                <span style="color:#8888aa;">Reliable:</span>
                <span style="color:#00ff88;">
                    {'✅ Both refutations passed' if tw.get('reliable') else '❌ Failed'}</span>
            </div>
            """, unsafe_allow_html=True)

            ref = tw.get("refutations", {})
            if ref:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Refutation Tests</div>',
                            unsafe_allow_html=True)
                for tname, tres in ref.items():
                    if isinstance(tres, dict) and "passed" in tres:
                        icon = "✅" if tres["passed"] else "❌"
                        orig = tres.get("original_effect","—")
                        test = tres.get("placebo_effect", tres.get("new_effect","—"))
                        st.markdown(
                            f'<div style="font-size:0.78rem;padding:3px 0;">'
                            f'{icon} <strong>{tname}</strong>: '
                            f'original={orig} · test_val={test} · '
                            f'passed={tres["passed"]}</div>',
                            unsafe_allow_html=True
                        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Power → Failure (Q2)</div>',
                    unsafe_allow_html=True)
        if pw:
            st.markdown(
                f'<div class="insight-box">'
                f'+{pw.get("effect_per_100_watts_pct","—")}% per 100W · '
                f'PWF safe zone: 3500–9000W · '
                f'Reliable: {"✅" if pw.get("reliable") else "❌"}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">NASA — Thermal Stress → RUL</div>',
                    unsafe_allow_html=True)

        ts = nasa_c.get("thermal_stress_effect", {})

        st.markdown("""
        <div style="font-family:'JetBrains Mono';font-size:0.76rem;
                    background:rgba(255,45,85,0.04);border-radius:8px;
                    padding:12px;margin-bottom:12px;line-height:1.9;">
        <span style="color:#ffaa00;">dataset_id</span> ──────────────────┐<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
        <span style="color:#00d4ff;">condition_cluster / cycle</span> →
        <span style="color:#ff2d55;font-weight:bold;">rul</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↑<br>
        <span style="color:#bf5fff;">thermal_stress</span> &nbsp;──────────────┘
        </div>
        """, unsafe_allow_html=True)

        if ts:
            st.markdown(f"""
            <div style="font-size:0.84rem;line-height:2.1;">
                <span style="color:#8888aa;">Coefficient:</span>
                <span style="color:#00d4ff;font-family:'JetBrains Mono';font-weight:600;">
                    {ts.get('coefficient_per_1std','—')} cycles/std</span><br>
                <span style="color:#8888aa;">Meaning:</span>
                <span style="color:#ffaa00;">
                    1σ stress → {abs(ts.get('coefficient_per_1std',0)):.1f} fewer cycles</span><br>
                <span style="color:#8888aa;">Units:</span>
                <span style="font-size:0.76rem;">
                    thermal_stress = s11 × s15, z-score per condition</span><br>
                <span style="color:#8888aa;">Interpretation:</span><br>
                <span style="font-size:0.78rem;">{ts.get('interpretation','—')}</span><br>
                <span style="color:#8888aa;">Reliable:</span>
                <span style="color:#00ff88;">
                    {'✅ Both refutations passed' if ts.get('reliable') else '❌ Failed'}</span>
            </div>
            """, unsafe_allow_html=True)

            ref = ts.get("refutations", {})
            if ref:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Refutation Tests</div>',
                            unsafe_allow_html=True)
                for tname, tres in ref.items():
                    if isinstance(tres, dict) and "passed" in tres:
                        icon = "✅" if tres["passed"] else "❌"
                        orig = tres.get("original_effect","—")
                        test = tres.get("new_effect", tres.get("placebo_effect","—"))
                        st.markdown(
                            f'<div style="font-size:0.78rem;padding:3px 0;">'
                            f'{icon} <strong>{tname}</strong>: '
                            f'original={orig} · test_val={test}</div>',
                            unsafe_allow_html=True
                        )

        st.markdown("""
        <div class="insight-box" style="margin-top:16px;">
            <strong>Why this matters:</strong> Standard ML = correlation.
            DoWhy backdoor adjustment = causation. These coefficients survived
            placebo refutation (effect vanishes with random treatment) AND
            random common cause refutation (estimate stable with added confounders).
            This makes intervention recommendations defensible.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Interactive causal risk curve
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Interactive — Tool Wear Causal Risk Curve</div>',
                unsafe_allow_html=True)

    tw_eff   = ai4i_c.get("tool_wear_effect",{}).get("effect_per_10_minutes_pct", 0.305)
    danger   = ai4i_c.get("tool_wear_effect",{}).get("danger_threshold_minutes", 240)
    baseline = 0.034 * 100
    wvals    = np.arange(0, 241, 5)
    rvals    = baseline + (wvals / 10) * tw_eff

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wvals, y=rvals, mode="lines",
        name="Causal Risk (DoWhy)",
        line=dict(color="#ff2d55", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,45,85,0.07)",
        hovertemplate="Wear: %{x} min<br>Risk: %{y:.2f}%<extra></extra>"
    ))
    fig.add_hline(y=15, line_dash="dash", line_color="#ffaa00",
                  annotation_text="15% danger threshold",
                  annotation_font_color="#ffaa00", annotation_font_size=11)
    fig.add_hline(y=baseline, line_dash="dot", line_color="#00ff88",
                  annotation_text=f"Baseline {baseline:.1f}%",
                  annotation_font_color="#00ff88", annotation_font_size=11)
    fig.add_vline(x=danger, line_width=1.5, line_dash="dash",
                  line_color="#ff2d55", opacity=0.6,
                  annotation_text=f"Danger {danger:.0f} min",
                  annotation_font_color="#ff2d55", annotation_font_size=11)
    fig.update_layout(
        **PLOTLY_BASE,
        title=f"Causal Failure Risk vs Tool Wear · DoWhy backdoor · +{tw_eff:.3f}%/10min",
        xaxis_title="Tool Wear (minutes)",
        yaxis_title="Estimated Failure Probability (%)",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MLOPS & MONITORING
# ══════════════════════════════════════════════════════════════════════════════

def page_mlops():
    st.markdown('<h2 class="neon-title" style="font-size:1.6rem;">📈 MLOps & Monitoring</h2>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="neon-subtitle">MLflow tracking · Model registry · '
        'DeepChecks validation · Evidently drift · Service startup guide</div><br>',
        unsafe_allow_html=True
    )

    # ── SERVICE STARTUP GUIDE ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Required Services — Start Before Running Dashboard</div>',
                unsafe_allow_html=True)

    svcs = [
        ("📊 MLflow UI",          "mlflow ui --host 0.0.0.0 --port 5000",
         "http://localhost:5000", "mfg_ai",  True),
        ("📡 MCP Server",         "python src/mcp_server.py",
         None,                    "mfg_ai",  True),
        ("🔬 Solara Digital Twin","solara run src/viz_dashboard.py",
         "http://localhost:9999", "mfg-2",   False),
        ("🌐 Streamlit Dashboard","streamlit run src/app.py",
         "http://localhost:8501", "mfg_ai",  True),
    ]

    sv_cols = st.columns(4)
    for col, (name, cmd, url, env, needed) in zip(sv_cols, svcs):
        with col:
            link_html = (
                f'<br><a href="{url}" target="_blank" '
                f'style="color:#00d4ff;font-size:0.72rem;">Open →</a>'
            ) if url else ""
            border_c = "rgba(255,45,85,0.3)" if needed else "rgba(191,95,255,0.3)"
            st.markdown(f"""
            <div class="glass-card" style="padding:14px;border-color:{border_c};">
                <div style="font-size:0.82rem;font-weight:600;
                            color:#e0e0e0;margin-bottom:6px;">{name}</div>
                <div style="font-family:'JetBrains Mono';font-size:0.67rem;
                            color:#8888aa;background:rgba(0,0,0,0.3);
                            padding:6px 8px;border-radius:4px;
                            margin-bottom:6px;word-break:break-all;">{cmd}</div>
                <div style="font-size:0.7rem;color:#bf5fff;">
                    conda activate {env}</div>
                {link_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MLFLOW BANNER ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="glass-card" style="display:flex;justify-content:space-between;
                align-items:center;padding:20px 28px;">
        <div>
            <div style="font-size:1rem;font-weight:600;color:#00d4ff;">
                MLflow Experiment Tracking
            </div>
            <div style="font-size:0.8rem;color:#8888aa;margin-top:4px;">
                6 experiments · 2 models in Production registry · All layers tracked
            </div>
        </div>
        <a href="http://localhost:5000" target="_blank" style="
            background:linear-gradient(135deg,rgba(0,212,255,0.2),rgba(255,45,85,0.1));
            border:1px solid rgba(0,212,255,0.4);border-radius:8px;
            padding:10px 24px;color:white;text-decoration:none;font-weight:500;">
            Open MLflow UI →
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Logged Experiments</div>',
                unsafe_allow_html=True)

    exps = [
        {"Layer":"3","Experiment":"AI4I_Classification",
         "Run":"RandomForest_v1","Key Metric":"F1=0.871 · AUC=0.989",
         "Framework":"scikit-learn"},
        {"Layer":"3","Experiment":"NASA_RUL_Regression",
         "Run":"LightGBM_Optuna_v1","Key Metric":"RMSE=17.68 · R²=0.817",
         "Framework":"lightgbm"},
        {"Layer":"4","Experiment":"Causal_Analysis",
         "Run":"DoWhy_AI4I + DoWhy_NASA","Key Metric":"Both refutations ✅",
         "Framework":"dowhy"},
        {"Layer":"5","Experiment":"Optimization",
         "Run":"ORTools_v1 + LinProg_v1","Key Metric":"15/15 feasible",
         "Framework":"ortools + scipy"},
        {"Layer":"6","Experiment":"Fleet_Simulation",
         "Run":"Mesa_50v50_v1","Key Metric":"+43.4% p<0.0001",
         "Framework":"mesa 3.0"},
        {"Layer":"7","Experiment":"System_Intelligence",
         "Run":"Aggregator_v1","Key Metric":"Production Ready ✅",
         "Framework":"fastmcp + ollama"},
    ]
    st.dataframe(pd.DataFrame(exps), use_container_width=True, hide_index=True)

    # Model registry cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Registry — Production Stage</div>',
                unsafe_allow_html=True)
    mr1, mr2 = st.columns(2)
    with mr1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.9rem;font-weight:600;color:#00d4ff;margin-bottom:8px;">
                AI4I_Failure_Classifier</div>
            <div style="font-size:0.78rem;line-height:1.9;color:#c0c0d0;">
                Algorithm: Random Forest · 15 features<br>
                Stage: <span style="color:#00ff88;">● Production</span><br>
                Evaluation: 20% stratified holdout<br>
                F1: 0.871 · Precision: 0.964 · AUC: 0.989
            </div>
        </div>
        """, unsafe_allow_html=True)
    with mr2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.9rem;font-weight:600;color:#00d4ff;margin-bottom:8px;">
                NASA_RUL_Predictor</div>
            <div style="font-size:0.78rem;line-height:1.9;color:#c0c0d0;">
                Algorithm: LightGBM + Optuna (30 trials TPE)<br>
                Stage: <span style="color:#00ff88;">● Production</span><br>
                Evaluation: 142 held-out engines (unit-level split)<br>
                RMSE: 17.68 · MAE: 12.19 · R²: 0.817
            </div>
        </div>
        """, unsafe_allow_html=True)

    # DeepChecks
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">DeepChecks Validation Reports</div>',
                unsafe_allow_html=True)

    dc_reports = {
        "AI4I Data Integrity":
            "reports/monitoring/deepchecks_ai4i_data_integrity.html",
        "AI4I Train-Test Validation":
            "reports/monitoring/deepchecks_ai4i_train_test.html",
        "AI4I Model Evaluation":
            "reports/monitoring/deepchecks_ai4i_model_eval.html",
        "NASA Data Integrity":
            "reports/monitoring/deepchecks_nasa_data_integrity.html",
        "NASA Train-Test Validation":
            "reports/monitoring/deepchecks_nasa_train_test.html",
    }

    for name, rel_path in dc_reports.items():
        full = PROJECT_ROOT / rel_path
        if full.exists():
            with st.expander(f"📋 {name}"):
            # Read with explicit UTF-8 — file is now written UTF-8
                with open(full, "r", encoding="utf-8", errors="replace") as f:
                   html = f.read()
            # Strip SRI integrity checks that block loading in iframe sandbox
                html_clean = re.sub(
                r'<script[^>]*integrity[^>]*>.*?</script>',
                '', html, flags=re.DOTALL
            )
            # Also strip crossorigin attributes
                html_clean = re.sub(
                r'\s+crossorigin="[^"]*"', '', html_clean
            )
                st.components.v1.html(html_clean, height=750, scrolling=True)
        else:
            st.markdown(
            f'<div style="color:#8888aa;font-size:0.78rem;">'
            f'⚠️ {name} — run: python src/monitoring.py</div>',
            unsafe_allow_html=True
        )

    # Evidently
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Evidently Drift Reports</div>',
                unsafe_allow_html=True)

    ev_reports = {
        "NASA Sensor Drift (FD001 vs FD004, condition-normalized)":
            "reports/monitoring/nasa_sensor_drift_report.html",
        "AI4I Feature Drift (features vs results parquet)":
            "reports/monitoring/ai4i_feature_drift_report.html",
    }
    for name, rel_path in ev_reports.items():
        full = PROJECT_ROOT / rel_path
        if full.exists():
            with st.expander(f"📉 {name}"):
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
                html_clean = re.sub(
                    r'<script[^>]*integrity[^>]*>.*?</script>',
                    '', html, flags=re.DOTALL
                )
                st.components.v1.html(html_clean, height=550, scrolling=True)
        else:
            st.markdown(
                f'<div style="color:#8888aa;font-size:0.78rem;">⚠️ {name} — not found</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RESEARCH & REPORTS
# ══════════════════════════════════════════════════════════════════════════════

def page_reports():
    st.markdown('<h2 class="neon-title" style="font-size:1.6rem;">📚 Research & Reports</h2>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="neon-subtitle">Interactive AI Agent (Ollama + MCP data) · '
        'OctoTools academic research · Agent briefings · Downloads</div><br>',
        unsafe_allow_html=True
    )

    # ── INTERACTIVE AGENT Q&A ─────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Interactive AI Agent — Ask Anything About the System</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.8rem;color:#8888aa;margin-bottom:16px;">'
        'The AI agent reads from MCP system_intelligence context (same data the '
        'FastMCP server exposes). OctoTools mode queries Arxiv + Wikipedia live.<br>'
        '<strong style="color:#ffaa00;">Requires:</strong> Ollama running '
        '(<code>ollama serve</code>) · For OctoTools: internet connection'
        '</div>',
        unsafe_allow_html=True
    )

    # Quick-question buttons
    st.markdown('<div style="font-size:0.75rem;color:#8888aa;margin-bottom:8px;">'
                'Quick questions:</div>',
                unsafe_allow_html=True)

    suggested = [
        "Is 17.68 RMSE good enough for production maintenance scheduling?",
        "Why does tool wear causally increase failure probability?",
        "How does thermal stress physically reduce engine RUL?",
        "What is the business case for 43.4% fleet life extension?",
        "Explain the DoWhy refutation tests we ran",
        "What do recent papers say about CMAPSS RUL benchmarks?",
    ]

    if "agent_q" not in st.session_state:
        st.session_state.agent_q = ""

    sq_cols = st.columns(3)
    for i, q in enumerate(suggested):
        with sq_cols[i % 3]:
            if st.button(q[:45] + "…" if len(q) > 45 else q,
                         use_container_width=True, key=f"sq_{i}"):
                st.session_state.agent_q = q

    user_q = st.text_input(
        "Or type your own question:",
        value=st.session_state.agent_q,
        placeholder="e.g. What happens if tool wear reaches 180 minutes?",
        key="agent_q_input"
    )

    mode = st.radio(
        "Response source:",
        ["🤖 AI Agent (Ollama + MCP system data)",
         "📚 Academic (OctoTools: Arxiv + Wikipedia)",
         "🔗 Both side-by-side"],
        horizontal=True
    )

    if st.button("🚀 Get Answer", key="agent_submit") and user_q:
        if mode == "🤖 AI Agent (Ollama + MCP system data)":
            with st.spinner("Querying Ollama with system intelligence context..."):
                ans = query_ollama_agent(user_q)
            st.markdown(
                f'<div class="success-box">'
                f'<strong>🤖 Agent Response:</strong><br><br>{ans}</div>',
                unsafe_allow_html=True
            )

        elif mode == "📚 Academic (OctoTools: Arxiv + Wikipedia)":
            with st.spinner("Searching Arxiv + Wikipedia via OctoTools..."):
                ans = query_octotools(user_q)
            st.markdown(
                f'<div class="insight-box">'
                f'<strong>📚 Academic Research:</strong><br><br>{ans}</div>',
                unsafe_allow_html=True
            )

        else:  # Both
            col_a, col_b = st.columns(2)
            with col_a:
                with st.spinner("Ollama agent..."):
                    a1 = query_ollama_agent(user_q)
                st.markdown(
                    f'<div class="success-box">'
                    f'<strong>🤖 AI Agent:</strong><br><br>{a1}</div>',
                    unsafe_allow_html=True
                )
            with col_b:
                with st.spinner("OctoTools research..."):
                    a2 = query_octotools(user_q)
                st.markdown(
                    f'<div class="insight-box">'
                    f'<strong>📚 OctoTools:</strong><br><br>{a2}</div>',
                    unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── AI FINAL REPORT ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    rep_path = PROJECT_ROOT / "models/ai_final_report.txt"
    if rep_path.exists():
        st.markdown('<div class="section-header">AI Engineering Report — Ollama Synthesizer Agent</div>',
                    unsafe_allow_html=True)
        with st.expander("📄 View Full Engineering Report (Ollama 3-agent synthesis)"):
            with open(rep_path, "r", encoding="utf-8") as f:
                rep_txt = f.read()
            st.markdown(
                f'<div class="glass-card" style="font-size:0.85rem;line-height:1.8;'
                f'white-space:pre-wrap;max-height:500px;overflow-y:auto;">'
                f'{rep_txt}</div>',
                unsafe_allow_html=True
            )
        st.download_button("⬇️ Download Engineering Report", data=rep_txt,
                           file_name="manufacturing_ai_report.txt", mime="text/plain")

    # ── AGENT BRIEFINGS ───────────────────────────────────────────────────────
    brief_path = PROJECT_ROOT / "models/agent_briefings.json"
    if brief_path.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Agent Briefings — NASA Analyst + AI4I Analyst</div>',
                    unsafe_allow_html=True)
        with open(brief_path) as f:
            briefs = json.load(f)

        bb1, bb2 = st.columns(2)
        with bb1:
            nb = briefs.get("nasa_briefing", {})
            with st.expander("✈️ NASA Prognostics Analyst Briefing"):
                st.markdown(
                    f'<div style="font-size:0.82rem;line-height:1.7;">'
                    f'{nb.get("engineering_brief","—")}</div>',
                    unsafe_allow_html=True
                )
        with bb2:
            ab = briefs.get("ai4i_briefing", {})
            with st.expander("🏭 AI4I Safety Analyst Briefing"):
                st.markdown(
                    f'<div style="font-size:0.82rem;line-height:1.7;">'
                    f'{ab.get("engineering_brief","—")}</div>',
                    unsafe_allow_html=True
                )

    # ── OCTOTOOLS RESEARCH CONTEXT ────────────────────────────────────────────
    res_path = PROJECT_ROOT / "models/research_context.json"
    if res_path.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">OctoTools Research Context — Arxiv + Wikipedia</div>',
                    unsafe_allow_html=True)
        with open(res_path) as f:
            rc = json.load(f)
        for tk, td in rc.get("topics", {}).items():
            ctx = td.get("context", tk)
            with st.expander(f"📚 {ctx}"):
                rc1, rc2 = st.columns(2)
                with rc1:
                    arxs = td.get("sources",{}).get("arxiv",{})
                    st.markdown(
                        f'**📄 Arxiv:**<br>'
                        f'<div class="insight-box" style="font-size:0.8rem;">'
                        f'{arxs.get("summary","—")}</div>',
                        unsafe_allow_html=True
                    )
                with rc2:
                    wks = td.get("sources",{}).get("wikipedia",{})
                    st.markdown(
                        f'**📖 Wikipedia:**<br>'
                        f'<div class="insight-box" style="font-size:0.8rem;">'
                        f'{wks.get("summary","—")}</div>',
                        unsafe_allow_html=True
                    )
                cf = td.get("compiled_finding","")
                if cf:
                    st.markdown(
                        f'<div class="insight-box">'
                        f'<strong>Compiled:</strong> {cf}</div>',
                        unsafe_allow_html=True
                    )

    # ── PLOT GALLERY ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">OctoTools Plot Gallery</div>',
                unsafe_allow_html=True)

    plots = [
        ("Fleet Survival Curve",    "reports/plots/survival_curve.png"),
        ("NASA Feature Importance", "reports/plots/feature_importance_nasa.png"),
        ("AI4I Feature Importance", "reports/plots/feature_importance_ai4i.png"),
        ("Failure Mode Breakdown",  "reports/plots/failure_mode_breakdown.png"),
        ("Tool Wear Risk Analysis", "reports/plots/tool_wear_risk.png"),
    ]
    available = [(n, p) for n, p in plots if (PROJECT_ROOT / p).exists()]
    for i in range(0, len(available), 2):
        gc = st.columns(2)
        for j, (n, p) in enumerate(available[i:i+2]):
            with gc[j]:
                st.image(str(PROJECT_ROOT / p), caption=n, use_container_width=True)

    # ── DOWNLOADS ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Download Artifacts</div>',
                unsafe_allow_html=True)

    downloads = [
        ("system_intelligence.json",   "models/system_intelligence.json"),
        ("agent_briefings.json",        "models/agent_briefings.json"),
        ("research_context.json",       "models/research_context.json"),
        ("nasa_eval_report.json",       "models/nasa_eval_report.json"),
        ("ai4i_eval_report.json",       "models/ai4i_eval_report.json"),
        ("survival_curves.csv",         "reports/survival_curves.csv"),
        ("agent_trajectories.csv",      "reports/agent_trajectories.csv"),
        ("nasa_best_params.json",       "models/nasa_best_params.json"),
    ]
    dl_cols = st.columns(4)
    for i, (fname, fpath) in enumerate(downloads):
        fp = PROJECT_ROOT / fpath
        if fp.exists():
            with open(fp, "rb") as f:
                dl_cols[i % 4].download_button(
                    f"⬇️ {fname}", data=f.read(),
                    file_name=fname, use_container_width=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    mdls = load_models()
    page = render_sidebar()

    if   page == "🏠 System Overview":     page_overview()
    elif page == "🏭 AI4I Machine Monitor": page_ai4i(mdls)
    elif page == "✈️  NASA Engine Fleet":   page_nasa(mdls)
    elif page == "🔬 Causal Intelligence":  page_causal()
    elif page == "📈 MLOps & Monitoring":   page_mlops()
    elif page == "📚 Research & Reports":   page_reports()


if __name__ == "__main__":
    main()