# src/mcp_server.py
import json
import os
from datetime import datetime
import polars as pl
import pandas as pd
from fastmcp import FastMCP

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR           = "models"
DATA_DIR            = "data/processed"
REPORTS_DIR         = "reports"

AI4I_EVAL_PATH      = f"{MODEL_DIR}/ai4i_eval_report.json"
NASA_EVAL_PATH      = f"{MODEL_DIR}/nasa_eval_report.json"
AI4I_CAUSAL_PATH    = f"{MODEL_DIR}/ai4i_causal_report.json"
NASA_CAUSAL_PATH    = f"{MODEL_DIR}/nasa_causal_report.json"
AI4I_OPT_PATH       = f"{MODEL_DIR}/ai4i_optimizer_results.json"
NASA_OPT_PATH       = f"{MODEL_DIR}/nasa_optimizer_results.json"
SIM_RESULTS_PATH    = f"{MODEL_DIR}/simulation_results.json"
SYSTEM_INTEL_PATH   = f"{MODEL_DIR}/system_intelligence.json"
AI4I_RESULTS_PATH   = f"{DATA_DIR}/ai4i_results.parquet"
NASA_FEATURES_PATH  = f"{DATA_DIR}/nasa_features.parquet"
SURVIVAL_CSV_PATH   = f"{REPORTS_DIR}/survival_curves.csv"

# ── helpers ────────────────────────────────────────────────────────────────────
def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}
    with open(path) as f:
        return json.load(f)


# ── FastMCP server ─────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="ManufacturingIntelligence",
    instructions=(
        "Manufacturing AI system intelligence server. "
        "Exposes model performance, causal findings, optimizer results, "
        "and simulation statistics from a predictive maintenance pipeline. "
        "Use resources to fetch structured data and tools for live queries."
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCES — read-only data the agents fetch
# ══════════════════════════════════════════════════════════════════════════════

@mcp.resource("manufacturing://nasa/model_performance")
def nasa_model_performance() -> str:
    """
    NASA RUL model performance metrics from held-out test engines.
    LightGBM model trained on CMAPSS dataset.
    Metrics: RMSE, MAE, R², evaluation method, top features.
    """
    data = _load_json(NASA_EVAL_PATH)
    return json.dumps(data, indent=2)


@mcp.resource("manufacturing://nasa/causal_analysis")
def nasa_causal_analysis() -> str:
    """
    DoWhy causal analysis results for NASA engine degradation.
    Primary finding: thermal stress effect on RUL.
    Includes refutation test results for reliability assessment.
    """
    data = _load_json(NASA_CAUSAL_PATH)
    return json.dumps(data, indent=2)


@mcp.resource("manufacturing://nasa/optimizer")
def nasa_optimizer() -> str:
    """
    NASA intervention optimizer results.
    Linear programming recommendations for thermal stress reduction.
    Three scenarios: critical (<20 RUL), degrading (20-50), healthy (>80).
    Includes both causal estimate and model-validated RUL gain.
    """
    data = _load_json(NASA_OPT_PATH)
    return json.dumps(data, indent=2)


@mcp.resource("manufacturing://nasa/simulation")
def nasa_simulation() -> str:
    """
    Mesa agent-based simulation results.
    50 AI-managed vs 50 unmanaged engines.
    Includes life extension %, statistical significance (Mann-Whitney U).
    """
    data = _load_json(SIM_RESULTS_PATH)
    return json.dumps(data, indent=2)


@mcp.resource("manufacturing://ai4i/model_performance")
def ai4i_model_performance() -> str:
    """
    AI4I failure classification model performance.
    Random Forest trained on 10,000 machine cycles.
    Metrics: F1, Precision, Recall, ROC-AUC, CV F1.
    Class imbalance: 3.4% failure rate, handled with class_weight=balanced.
    """
    data = _load_json(AI4I_EVAL_PATH)
    return json.dumps(data, indent=2)


@mcp.resource("manufacturing://ai4i/causal_analysis")
def ai4i_causal_analysis() -> str:
    """
    DoWhy causal analysis for AI4I machine failures.
    Q1: Tool wear effect on failure probability (per 10 minutes).
    Q2: Power consumption effect on failure probability (per 100W).
    Includes danger threshold and refutation results.
    """
    data = _load_json(AI4I_CAUSAL_PATH)
    return json.dumps(data, indent=2)


@mcp.resource("manufacturing://ai4i/optimizer")
def ai4i_optimizer() -> str:
    """
    OR-Tools CP-SAT optimizer results for AI4I machine setpoints.
    Optimal RPM and Torque per tool wear scenario and product type (L/M/H).
    Constraints: PWF safe range, OSF threshold, TWF onset, causal danger threshold.
    """
    data = _load_json(AI4I_OPT_PATH)
    return json.dumps(data, indent=2)


@mcp.resource("manufacturing://ai4i/anomaly_summary")
def ai4i_anomaly_summary() -> str:
    """
    Summary statistics from AI4I results parquet.
    Includes: total samples, failure count, anomaly count,
    failure rate, anomaly rate, avg failure probability.
    Source: live computation from ai4i_results.parquet.
    """
    if not os.path.exists(AI4I_RESULTS_PATH):
        return json.dumps({"error": "ai4i_results.parquet not found"})

    df = pl.read_parquet(AI4I_RESULTS_PATH)
    total    = len(df)
    failures = int(df["failure"].sum())
    anomalies = int(df["is_anomaly"].sum())

    summary = {
        "total_samples":       total,
        "failure_count":       failures,
        "failure_rate_pct":    round(failures / total * 100, 2),
        "anomaly_count":       anomalies,
        "anomaly_rate_pct":    round(anomalies / total * 100, 2),
        "avg_failure_prob":    round(float(df["failure_prob"].mean()), 4),
        "high_risk_count":     int((df["failure_prob"] > 0.5).sum()),
        "failure_modes": {
            "TWF": int(df["TWF"].sum()),
            "HDF": int(df["HDF"].sum()),
            "PWF": int(df["PWF"].sum()),
            "OSF": int(df["OSF"].sum()),
            "RNF": int(df["RNF"].sum()),
        }
    }
    return json.dumps(summary, indent=2)


@mcp.resource("manufacturing://system/survival_summary")
def system_survival_summary() -> str:
    """
    Fleet survival curve summary statistics.
    Computed from Mesa simulation survival_curves.csv.
    Includes: cycle at which 50% of each fleet survives.
    """
    if not os.path.exists(SURVIVAL_CSV_PATH):
        return json.dumps({"error": "survival_curves.csv not found"})

    df = pd.read_csv(SURVIVAL_CSV_PATH)

    def half_life(col):
        below = df[df[col] <= 50]
        return int(below["cycle"].iloc[0]) if len(below) > 0 else None

    summary = {
        "total_cycles_simulated": int(df["cycle"].max()),
        "managed_half_life_cycle":   half_life("survival_pct_managed"),
        "unmanaged_half_life_cycle": half_life("survival_pct_unmanaged"),
        "managed_final_survival_pct":   round(
            float(df["survival_pct_managed"].iloc[-1]), 1),
        "unmanaged_final_survival_pct": round(
            float(df["survival_pct_unmanaged"].iloc[-1]), 1),
    }
    return json.dumps(summary, indent=2)


@mcp.resource("manufacturing://system/full_intelligence")
def system_full_intelligence() -> str:
    """
    Complete compiled system intelligence from all pipeline layers.
    Single source of truth for the manufacturing AI system.
    Includes all model metrics, causal findings, optimizer results,
    simulation statistics, and system status checks.
    """
    data = _load_json(SYSTEM_INTEL_PATH)
    return json.dumps(data, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS — live query functions agents can call
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_ai4i_statistics() -> str:
    """
    Run live statistical queries on the AI4I results dataset.
    Returns failure and anomaly statistics, feature correlations,
    and failure mode breakdown.
    Use this for precise numbers not in the pre-compiled summary.
    """
    if not os.path.exists(AI4I_RESULTS_PATH):
        return json.dumps({"error": "ai4i_results.parquet not found"})

    df = pl.read_parquet(AI4I_RESULTS_PATH)

    # Failure probability distribution
    fp = df["failure_prob"]
    stats = {
        "failure_prob_stats": {
            "mean":   round(float(fp.mean()), 4),
            "median": round(float(fp.median()), 4),
            "p75":    round(float(fp.quantile(0.75)), 4),
            "p90":    round(float(fp.quantile(0.90)), 4),
            "p95":    round(float(fp.quantile(0.95)), 4),
            "p99":    round(float(fp.quantile(0.99)), 4),
        },
        "tool_wear_stats": {
            "mean":    round(float(df["tool_wear"].mean()), 1),
            "median":  round(float(df["tool_wear"].median()), 1),
            "max":     round(float(df["tool_wear"].max()), 1),
        },
        "power_stats": {
            "mean":    round(float(df["power_w"].mean()), 1),
            "median":  round(float(df["power_w"].median()), 1),
        },
        "anomaly_vs_failure_overlap": int(
            ((df["is_anomaly"] == 1) & (df["failure"] == 1)).sum()
        ),
        "anomaly_without_failure": int(
            ((df["is_anomaly"] == 1) & (df["failure"] == 0)).sum()
        ),
        "failure_without_anomaly": int(
            ((df["is_anomaly"] == 0) & (df["failure"] == 1)).sum()
        ),
    }
    return json.dumps(stats, indent=2)


@mcp.tool()
def get_nasa_statistics() -> str:
    """
    Run live statistical queries on the NASA features dataset.
    Returns RUL distribution, thermal stress statistics by dataset,
    and engine count per sub-dataset.
    Use this for precise fleet-level numbers.
    """
    if not os.path.exists(NASA_FEATURES_PATH):
        return json.dumps({"error": "nasa_features.parquet not found"})

    df = pl.read_parquet(NASA_FEATURES_PATH)

    rul_stats = {
        "overall": {
            "mean":   round(float(df["rul"].mean()), 1),
            "median": round(float(df["rul"].median()), 1),
            "min":    round(float(df["rul"].min()), 1),
            "max":    round(float(df["rul"].max()), 1),
        }
    }

    # Per dataset stats
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        sub = df.filter(pl.col("dataset_id") == ds)
        if len(sub) > 0:
            rul_stats[ds] = {
                "engines":       int(sub["unit"].n_unique()),
                "rows":          len(sub),
                "mean_rul":      round(float(sub["rul"].mean()), 1),
                "critical_count": int((sub["rul"] < 20).sum()),
            }

    thermal_stats = {
        "mean":  round(float(df["thermal_stress"].mean()), 4),
        "std":   round(float(df["thermal_stress"].std()), 4),
        "p90":   round(float(df["thermal_stress"].quantile(0.90)), 4),
        "p95":   round(float(df["thermal_stress"].quantile(0.95)), 4),
    }

    return json.dumps({
        "rul_statistics":     rul_stats,
        "thermal_stress":     thermal_stats,
        "total_engines":      int(df["unit"].n_unique()),
        "total_cycles":       len(df),
    }, indent=2)


@mcp.tool()
def get_system_status() -> str:
    """
    Check system readiness — all pipeline files present,
    model metrics meet minimum thresholds for production deployment.
    Returns overall ready_for_production bool and detail map.
    """
    required_files = {
        "nasa_eval_report":    NASA_EVAL_PATH,
        "ai4i_eval_report":    AI4I_EVAL_PATH,
        "nasa_causal_report":  NASA_CAUSAL_PATH,
        "ai4i_causal_report":  AI4I_CAUSAL_PATH,
        "nasa_optimizer":      NASA_OPT_PATH,
        "ai4i_optimizer":      AI4I_OPT_PATH,
        "simulation_results":  SIM_RESULTS_PATH,
        "system_intelligence": SYSTEM_INTEL_PATH,
        "ai4i_results":        AI4I_RESULTS_PATH,
        "nasa_features":       NASA_FEATURES_PATH,
    }

    file_status = {k: os.path.exists(v) for k, v in required_files.items()}
    all_present  = all(file_status.values())

    # Metric checks
    metric_checks = {}
    try:
        nasa_eval = _load_json(NASA_EVAL_PATH)
        metric_checks["nasa_r2_above_0.7"] = nasa_eval.get("r2", 0) >= 0.7
        metric_checks["nasa_rmse_below_25"] = nasa_eval.get("rmse", 999) < 25
    except Exception:
        metric_checks["nasa_metrics"] = False

    try:
        ai4i_eval = _load_json(AI4I_EVAL_PATH)
        metric_checks["ai4i_f1_above_0.6"] = ai4i_eval.get("f1_score", 0) >= 0.6
        metric_checks["ai4i_precision_above_0.8"] = (
            ai4i_eval.get("precision", 0) >= 0.8
        )
    except Exception:
        metric_checks["ai4i_metrics"] = False

    try:
        sim = _load_json(SIM_RESULTS_PATH)
        metric_checks["simulation_significant"] = (
            sim.get("statistical_significance", {}).get("significant", False)
        )
    except Exception:
        metric_checks["simulation_check"] = False

    all_metrics_pass = all(metric_checks.values())

    return json.dumps({
        "file_status":        file_status,
        "metric_checks":      metric_checks,
        "all_files_present":  all_present,
        "all_metrics_pass":   all_metrics_pass,
        "ready_for_production": all_present and all_metrics_pass,
        "checked_at":         datetime.now().isoformat(),
    }, indent=2)


@mcp.tool()
def get_engineer_setpoint(tool_wear: int, prod_type: str) -> str:
    """
    Get optimal machine setpoint for given tool wear and product type.

    Args:
        tool_wear: Current tool wear in minutes (0-240)
        prod_type: Product type — 'L', 'M', or 'H'

    Returns optimal RPM, Torque, Power and safety margins.
    Source: OR-Tools CP-SAT optimizer results.
    """
    data = _load_json(AI4I_OPT_PATH)
    pt   = prod_type.upper()

    if pt not in data:
        return json.dumps({"error": f"prod_type must be L, M, or H. Got: {prod_type}"})

    # Find closest scenario
    scenarios = {
        "New tool":              0,
        "Mid-life tool":         100,
        "Approaching TWF onset": 180,
        "Past TWF onset":        210,
        "Near end of life":      235,
    }

    closest = min(scenarios.items(), key=lambda x: abs(x[1] - tool_wear))
    scenario_name = closest[0]
    result = data[pt].get(scenario_name, {})
    result["matched_scenario"] = scenario_name
    result["requested_tool_wear"] = tool_wear

    return json.dumps(result, indent=2)


@mcp.tool()
def get_rul_intervention(scenario: str) -> str:
    """
    Get NASA engine RUL intervention recommendation for given scenario.

    Args:
        scenario: 'critical' (RUL<20), 'degrading' (RUL 20-50),
                  or 'healthy' (RUL>80)

    Returns thermal stress reduction recommendation and expected RUL gain.
    Source: SciPy linprog + DoWhy causal coefficient.
    """
    data = _load_json(NASA_OPT_PATH)
    s    = scenario.lower()

    if s not in data:
        return json.dumps({
            "error": f"scenario must be critical, degrading, or healthy. Got: {scenario}"
        })

    return json.dumps(data[s], indent=2)


# ── run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Manufacturing Intelligence MCP Server starting...")
    print("   Resources: 8 data endpoints")
    print("   Tools:     5 live query functions")
    print("   Protocol:  FastMCP over stdio")
    print("   Ready for agent connections.")
    mcp.run()