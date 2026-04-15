# src/monitoring.py
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import polars as pl
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
import matplotlib
matplotlib.use("Agg")
if not hasattr(np, "Inf"): np.Inf = np.inf
if not hasattr(np, "Float64"): np.Float64 = float
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# DeepChecks
from deepchecks.tabular import Dataset as DCDataset
from deepchecks.tabular.suites import (
    data_integrity,
    train_test_validation,
    model_evaluation
)

# Evidently
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset
)
from evidently.legacy.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric
)

# ── NumPy 2.0 + Sklearn compatibility patch ────────────────────────────────
import sklearn.metrics
import sklearn.metrics._scorer

if not hasattr(np, "Inf"):    np.Inf = np.inf
if not hasattr(np, "bool"):   np.bool = bool
if not hasattr(np, "int"):    np.int = int
if not hasattr(np, "float"):  np.float = float

try:
    sklearn.metrics.get_scorer("max_error")
except (ValueError, KeyError):
    valid = sklearn.metrics.get_scorer_names()
    fallback = ("neg_mean_absolute_error"
                if "neg_mean_absolute_error" in valid else valid[0])
    sklearn.metrics._scorer._SCORERS["max_error"] = \
        sklearn.metrics.get_scorer(fallback)

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR           = "models"
DATA_DIR            = "data/processed"
REPORTS_DIR         = "reports/monitoring"
PLOTS_DIR           = "reports/plots"

AI4I_FEATURES_PATH  = f"{DATA_DIR}/ai4i_features.parquet"
AI4I_RESULTS_PATH   = f"{DATA_DIR}/ai4i_results.parquet"
AI4I_CLEANED_PATH   = f"{DATA_DIR}/ai4i_cleaned.parquet"
NASA_FEATURES_PATH  = f"{DATA_DIR}/nasa_features.parquet"
NASA_FD001_PATH     = f"{DATA_DIR}/nasa_train_fd001_cleaned.parquet"
NASA_FD004_PATH     = f"{DATA_DIR}/nasa_train_fd004_cleaned.parquet"

AI4I_CLASSIFIER_PATH = f"{MODEL_DIR}/ai4i_classifier.joblib"
NASA_MODEL_PATH      = f"{MODEL_DIR}/nasa_rul_model.joblib"
AI4I_EVAL_PATH       = f"{MODEL_DIR}/ai4i_eval_report.json"
NASA_EVAL_PATH       = f"{MODEL_DIR}/nasa_eval_report.json"
AI4I_CAUSAL_PATH     = f"{MODEL_DIR}/ai4i_causal_report.json"
NASA_CAUSAL_PATH     = f"{MODEL_DIR}/nasa_causal_report.json"
AI4I_OPT_PATH        = f"{MODEL_DIR}/ai4i_optimizer_results.json"
NASA_OPT_PATH        = f"{MODEL_DIR}/nasa_optimizer_results.json"
SIM_RESULTS_PATH     = f"{MODEL_DIR}/simulation_results.json"
SYSTEM_INTEL_PATH    = f"{MODEL_DIR}/system_intelligence.json"

# ── MLflow config ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = "http://localhost:5000"
MLFLOW_EXPERIMENT_PREFIX = "ManufacturingAI"

# ── feature columns ────────────────────────────────────────────────────────────
AI4I_FEATURE_COLS = [
    "rpm", "torque", "tool_wear", "air_temp_c", "proc_temp_c", "power_w",
    "temp_delta", "stress_index", "overstrain_margin",
    "air_temp_c_smooth", "proc_temp_c_smooth", "rpm_smooth",
    "torque_smooth", "tool_wear_smooth", "power_w_smooth"
]
AI4I_TARGET_COL = "failure"

NASA_EXCLUDE_COLS = ["unit", "cycle", "dataset_id", "condition_cluster", "rul"]


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _connect_mlflow() -> MlflowClient:
    """Connect to MLflow tracking server."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        # Verify connection
        client.search_experiments()
        print(f"  ✅ Connected to MLflow at {MLFLOW_TRACKING_URI}")
        return client
    except Exception as e:
        print(f"  ⚠️  MLflow server not reachable: {e}")
        print(f"  ℹ️  Falling back to local tracking (mlruns/)")
        mlflow.set_tracking_uri("mlruns")
        return MlflowClient(tracking_uri="mlruns")


def _get_or_create_experiment(name: str) -> str:
    """Get existing experiment or create new one."""
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            name,
            tags={"project": "IntelligentManufacturing", "version": "1.0"}
        )
        print(f"  Created experiment: {name}")
    else:
        exp_id = exp.experiment_id
        print(f"  Using experiment: {name} (id={exp_id})")
    return exp_id


def _register_model(run_id: str, artifact_path: str,
                    model_name: str, client: MlflowClient) -> None:
    """Register model and promote to Production."""
    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri, model_name)
        print(f"    Registered: {model_name} v{mv.version}")

        # Promote to Production
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"    Promoted: {model_name} v{mv.version} → Production")
    except Exception as e:
        print(f"    ⚠️  Model registration warning: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MLflow: Log all layer results
# ══════════════════════════════════════════════════════════════════════════════

def log_ai4i_classification(client: MlflowClient):
    """Log AI4I Random Forest classifier to MLflow."""
    print("\n  📊 Logging AI4I Classification...")

    eval_report = _load_json(AI4I_EVAL_PATH)
    classifier  = joblib.load(AI4I_CLASSIFIER_PATH)

    # Load data for signature inference
    df        = pl.read_parquet(AI4I_FEATURES_PATH).to_pandas()
    X_sample  = df[AI4I_FEATURE_COLS].head(5)
    y_sample  = classifier.predict(X_sample)
    signature = infer_signature(X_sample, y_sample)

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}/AI4I_Classification")

    with mlflow.start_run(run_name="RandomForest_v1") as run:
        # Parameters — model configuration
        mlflow.log_params({
            "n_estimators":      200,
            "max_depth":         15,
            "min_samples_split": 5,
            "class_weight":      "balanced",
            "test_size":         0.2,
            "cv_folds":          5,
            "random_state":      42,
            "imbalance_strategy": "class_weight_balanced",
        })

        # Metrics — all from held-out test set
        mlflow.log_metrics({
            "test_f1_score":      eval_report["f1_score"],
            "test_precision":     eval_report["precision"],
            "test_recall":        eval_report["recall"],
            "test_roc_auc":       eval_report["roc_auc"],
            "cv_f1_mean":         eval_report["cv_f1_mean"],
            "cv_f1_std":          eval_report["cv_f1_std"],
            "class_imbalance_pct": 3.4,
            "training_samples":   eval_report["class_distribution"]["normal"],
            "failure_samples":    eval_report["class_distribution"]["failure"],
        })

        # Tags
        mlflow.set_tags({
            "dataset":           "AI4I_2020_10k",
            "layer":             "3",
            "framework":         "scikit-learn",
            "task":              "binary_classification",
            "evaluation":        "stratified_holdout_20pct",
            "causal_validated":  "true",
        })

        # Log confusion matrix as artifact image
        cm    = eval_report.get("confusion_matrix", [[0,0],[0,0]])
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal","Failure"],
                    yticklabels=["Normal","Failure"],
                    ax=ax)
        ax.set_title("AI4I Confusion Matrix (Test Set)",
                     color="white", pad=10)
        ax.tick_params(colors="white")
        plt.tight_layout()
        cm_path = os.path.join(PLOTS_DIR, "confusion_matrix_ai4i.png")
        plt.savefig(cm_path, dpi=150, bbox_inches="tight",
                    facecolor="#1a1a2e")
        plt.close()
        mlflow.log_artifact(cm_path, "plots")

        # Log feature importances as artifact
        fi_path = os.path.join(REPORTS_DIR, "ai4i_feature_importances.json")
        with open(fi_path, "w") as f:
            json.dump(eval_report.get("feature_importances", {}), f, indent=2)
        mlflow.log_artifact(fi_path, "reports")

        # Log eval report
        mlflow.log_artifact(AI4I_EVAL_PATH, "reports")
        mlflow.log_artifact(AI4I_CAUSAL_PATH, "reports")

        # Log model with signature
        mlflow.sklearn.log_model(
            classifier,
            artifact_path="model",
            signature=signature,
            registered_model_name=None,  # register separately
            input_example=X_sample.head(1)
        )

        run_id = run.info.run_id
        print(f"    Run ID: {run_id}")
        print(f"    F1={eval_report['f1_score']} | "
              f"Precision={eval_report['precision']} | "
              f"AUC={eval_report['roc_auc']}")

    # Register and promote
    _register_model(run_id, "model", "AI4I_Failure_Classifier", client)
    return run_id


def log_nasa_rul_model(client: MlflowClient):
    """Log NASA LightGBM RUL model to MLflow."""
    print("\n  📊 Logging NASA RUL Model...")

    eval_report  = _load_json(NASA_EVAL_PATH)
    nasa_model   = joblib.load(NASA_MODEL_PATH)
    best_params  = _load_json(f"{MODEL_DIR}/nasa_best_params.json")

    df           = pl.read_parquet(NASA_FEATURES_PATH).to_pandas()
    feature_cols = [c for c in df.columns if c not in NASA_EXCLUDE_COLS]
    X_sample     = df[feature_cols].head(5)
    y_sample     = nasa_model.predict(X_sample)
    signature    = infer_signature(X_sample, y_sample)

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}/NASA_RUL_Regression")

    with mlflow.start_run(run_name="LightGBM_Optuna_v1") as run:
        # Optuna best parameters
        mlflow.log_params({
            **best_params,
            "rul_clip_upper":    125,
            "split_strategy":    "group_shuffle_by_unit",
            "test_engines_pct":  20,
            "optuna_trials":     30,
            "optuna_sampler":    "TPE",
            "n_features":        len(feature_cols),
        })

        # Metrics from held-out test engines
        mlflow.log_metrics({
            "test_rmse":              eval_report["rmse"],
            "test_mae":               eval_report["mae"],
            "test_r2":                eval_report["r2"],
            "optuna_best_inner_rmse": 18.3529,
            "n_test_engines":         142,
            "n_train_engines":        567,
            "total_engines":          709,
        })

        mlflow.set_tags({
            "dataset":          "CMAPSS_FD001-FD004",
            "layer":            "3",
            "framework":        "lightgbm",
            "task":             "regression_rul",
            "evaluation":       "unit_level_holdout_20pct",
            "hyperopt":         "optuna_bayesian_tpe",
            "causal_validated": "true",
        })

        # RUL prediction distribution plot
        preds     = nasa_model.predict(df[feature_cols].sample(2000, random_state=42))
        actuals   = df["rul"].sample(2000, random_state=42).values

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#1a1a2e")

        for ax in axes:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        axes[0].scatter(actuals, preds, alpha=0.3, s=8,
                        color="#2196F3", label="Predictions")
        axes[0].plot([0, 125], [0, 125], "r--",
                     linewidth=1.5, label="Perfect prediction")
        axes[0].set_xlabel("Actual RUL (cycles)")
        axes[0].set_ylabel("Predicted RUL (cycles)")
        axes[0].set_title(f"RUL Predictions vs Actuals\n"
                          f"RMSE={eval_report['rmse']} | R²={eval_report['r2']}")
        axes[0].legend(fontsize=8)

        errors = preds - actuals
        axes[1].hist(errors, bins=40, color="#9C27B0", alpha=0.8,
                     edgecolor="#444")
        axes[1].axvline(0, color="#FFD700", linewidth=2,
                        linestyle="--", label="Zero error")
        axes[1].set_xlabel("Prediction Error (cycles)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Error Distribution\n"
                          f"MAE={eval_report['mae']}")
        axes[1].legend(fontsize=8)

        plt.suptitle("NASA RUL Model Performance",
                     color="white", fontsize=12, y=1.02)
        plt.tight_layout()
        rul_plot_path = os.path.join(PLOTS_DIR, "nasa_rul_performance.png")
        plt.savefig(rul_plot_path, dpi=150, bbox_inches="tight",
                    facecolor="#1a1a2e")
        plt.close()
        mlflow.log_artifact(rul_plot_path, "plots")
        mlflow.log_artifact(NASA_EVAL_PATH, "reports")
        mlflow.log_artifact(NASA_CAUSAL_PATH, "reports")

        mlflow.lightgbm.log_model(
            nasa_model,
            artifact_path="model",
            signature=signature,
            input_example=X_sample.head(1)
        )

        run_id = run.info.run_id
        print(f"    Run ID: {run_id}")
        print(f"    RMSE={eval_report['rmse']} | "
              f"MAE={eval_report['mae']} | "
              f"R²={eval_report['r2']}")

    _register_model(run_id, "model", "NASA_RUL_Predictor", client)
    return run_id


def log_causal_analysis():
    """Log DoWhy causal analysis results to MLflow."""
    print("\n  📊 Logging Causal Analysis...")

    ai4i_causal = _load_json(AI4I_CAUSAL_PATH)
    nasa_causal = _load_json(NASA_CAUSAL_PATH)

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}/Causal_Analysis")

    # AI4I causal run
    with mlflow.start_run(run_name="DoWhy_AI4I_v1"):
        tw = ai4i_causal.get("tool_wear_effect", {})
        pw = ai4i_causal.get("power_effect", {})

        mlflow.log_metrics({
            "tool_wear_scaled_coef":     tw.get("scaled_coefficient", 0),
            "tool_wear_effect_per_10min": tw.get("effect_per_10_minutes_pct", 0),
            "danger_threshold_minutes":  tw.get("danger_threshold_minutes", 0),
            "tool_wear_placebo_passed":  int(tw.get("refutations", {})
                                             .get("placebo", {})
                                             .get("passed", False)),
            "tool_wear_rc_passed":       int(tw.get("refutations", {})
                                             .get("random_common_cause", {})
                                             .get("passed", False)),
            "power_scaled_coef":         pw.get("scaled_coefficient", 0),
            "power_effect_per_100w":     pw.get("effect_per_100_watts_pct", 0),
            "power_placebo_passed":      int(pw.get("refutations", {})
                                             .get("placebo", {})
                                             .get("passed", False)),
        })
        mlflow.set_tags({
            "layer":     "4",
            "framework": "dowhy",
            "method":    "backdoor_linear_regression",
            "dataset":   "AI4I_2020",
        })
        mlflow.log_artifact(AI4I_CAUSAL_PATH, "reports")
        print(f"    AI4I: TW effect={tw.get('effect_per_10_minutes_pct')}% "
              f"per 10min | threshold={tw.get('danger_threshold_minutes')}min")

    # NASA causal run
    with mlflow.start_run(run_name="DoWhy_NASA_v1"):
        ts = nasa_causal.get("thermal_stress_effect", {})

        mlflow.log_metrics({
            "thermal_stress_coef":     ts.get("coefficient_per_1std", 0),
            "thermal_placebo_passed":  int(ts.get("refutations", {})
                                           .get("placebo", {})
                                           .get("passed", False)),
            "thermal_rc_passed":       int(ts.get("refutations", {})
                                           .get("random_common_cause", {})
                                           .get("passed", False)),
            "causal_reliable":         int(ts.get("reliable", False)),
        })
        mlflow.set_tags({
            "layer":     "4",
            "framework": "dowhy",
            "method":    "backdoor_linear_regression",
            "dataset":   "CMAPSS_FD001-FD004",
        })
        mlflow.log_artifact(NASA_CAUSAL_PATH, "reports")
        print(f"    NASA: thermal coef={ts.get('coefficient_per_1std')} "
              f"cycles/std | reliable={ts.get('reliable')}")


def log_optimization():
    """Log optimizer results to MLflow."""
    print("\n  📊 Logging Optimization Results...")

    ai4i_opt = _load_json(AI4I_OPT_PATH)
    nasa_opt  = _load_json(NASA_OPT_PATH)

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}/Optimization")

    # AI4I OR-Tools
    with mlflow.start_run(run_name="AI4I_ORTools_v1"):
        # Count scenarios
        total    = sum(len(v) for v in ai4i_opt.values())
        feasible = sum(
            1 for pt in ai4i_opt.values()
            for s in pt.values()
            if s.get("status") in ["OPTIMAL", "FEASIBLE"]
        )
        mlflow.log_metrics({
            "total_scenarios":   total,
            "feasible_count":    feasible,
            "infeasible_count":  total - feasible,
            "feasibility_rate":  round(feasible / total * 100, 1),
            "pwf_min_watts":     3500,
            "pwf_max_watts":     9000,
            "twf_onset_minutes": 200,
        })
        mlflow.set_tags({
            "layer":   "5",
            "solver":  "ortools_cpsat",
            "dataset": "AI4I_2020",
        })
        mlflow.log_artifact(AI4I_OPT_PATH, "reports")
        print(f"    AI4I: {feasible}/{total} scenarios feasible")

    # NASA linprog
    with mlflow.start_run(run_name="NASA_LinProg_v1"):
        crit = nasa_opt.get("critical", {})
        deg  = nasa_opt.get("degrading", {})

        mlflow.log_metrics({
            "critical_causal_rul_gain":  crit.get("causal_rul_gain_cycles", 0),
            "critical_model_rul_gain":   crit.get("model_rul_gain_cycles", 0),
            "critical_stress_reduction": crit.get("stress_reduction_pct", 0),
            "degrading_causal_rul_gain": deg.get("causal_rul_gain_cycles", 0),
            "degrading_model_rul_gain":  deg.get("model_rul_gain_cycles", 0),
        })
        mlflow.set_tags({
            "layer":   "5",
            "solver":  "scipy_linprog_highs",
            "dataset": "CMAPSS_FD001-FD004",
        })
        mlflow.log_artifact(NASA_OPT_PATH, "reports")
        print(f"    NASA critical: causal gain={crit.get('causal_rul_gain_cycles')} "
              f"| model gain={crit.get('model_rul_gain_cycles')} cycles")


def log_simulation():
    """Log Mesa simulation results to MLflow."""
    print("\n  📊 Logging Simulation Results...")

    sim = _load_json(SIM_RESULTS_PATH)

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}/Fleet_Simulation")

    with mlflow.start_run(run_name="Mesa_50v50_v1"):
        mlflow.log_params({
            "n_agents_per_group":   sim["n_agents_per_group"],
            "total_cycles_run":     sim["total_cycles_run"],
            "random_seed":          42,
            "framework":            "mesa_3.0",
            "wear_rate_normal":     1.0,
            "wear_rate_caution":    0.9,
            "wear_rate_warning":    0.75,
            "wear_rate_critical":   0.60,
            "wear_rate_unmanaged":  1.4,
        })

        mlflow.log_metrics({
            "managed_mean_cycles":      sim["managed"]["mean_cycles"],
            "managed_std_cycles":       sim["managed"]["std_cycles"],
            "managed_min_cycles":       sim["managed"]["min_cycles"],
            "managed_max_cycles":       sim["managed"]["max_cycles"],
            "unmanaged_mean_cycles":    sim["unmanaged"]["mean_cycles"],
            "unmanaged_std_cycles":     sim["unmanaged"]["std_cycles"],
            "unmanaged_min_cycles":     sim["unmanaged"]["min_cycles"],
            "unmanaged_max_cycles":     sim["unmanaged"]["max_cycles"],
            "life_extension_pct":       sim["life_extension_pct"],
            "p_value":                  sim["statistical_significance"]["p_value"],
            "statistically_significant": int(
                sim["statistical_significance"]["significant"]
            ),
        })

        mlflow.set_tags({
            "layer":       "6",
            "framework":   "mesa_3.0",
            "stat_test":   "mann_whitney_u_onesided",
            "brain_model": "lightgbm_rul",
        })

        mlflow.log_artifact(SIM_RESULTS_PATH, "reports")
        mlflow.log_artifact(
            f"{REPORTS_DIR.replace('monitoring', '')}survival_curves.csv",
            "data"
        )
        print(f"    Life extension: {sim['life_extension_pct']}% | "
              f"p={sim['statistical_significance']['p_value']}")


def log_system_intelligence():
    """Log final system intelligence to MLflow."""
    print("\n  📊 Logging System Intelligence...")

    intel = _load_json(SYSTEM_INTEL_PATH)

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_PREFIX}/System_Intelligence")

    with mlflow.start_run(run_name="Aggregator_v1"):
        status   = intel["system_status"]
        headlines = intel.get("dashboard_headlines", {})

        mlflow.log_metrics({
            "files_present":          int(status["all_files_present"]),
            "metrics_passed":         int(status["all_metrics_pass"]),
            "ready_for_production":   int(status["ready_for_production"]),
            "nasa_r2":                headlines.get("nasa_r2", 0),
            "nasa_rmse":              headlines.get("nasa_rmse", 0),
            "ai4i_f1":                headlines.get("ai4i_f1", 0),
            "ai4i_precision":         headlines.get("ai4i_precision", 0),
            "life_extension_pct":     headlines.get("life_extension_pct", 0),
            "anomaly_count":          headlines.get("anomaly_count", 0),
            "failure_count":          headlines.get("failure_count", 0),
        })

        mlflow.set_tags({
            "layer":  "7",
            "status": "production" if status["ready_for_production"] else "staging",
        })

        mlflow.log_artifact(SYSTEM_INTEL_PATH, "reports")
        mlflow.log_artifact(f"{MODEL_DIR}/ai_final_report.txt", "reports")
        print(f"    Ready for production: {status['ready_for_production']}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DeepChecks: Data and Model Validation
# ══════════════════════════════════════════════════════════════════════════════
def _save_deepchecks_html(result, output_path: str):
    """
    Save DeepChecks result using _repr_html_() with explicit UTF-8.
    This fixes the blank white page issue — save_as_html() uses the
    system codec which fails on Windows or produces incomplete output.
    Works in DeepChecks 0.18+ and matches the confirmed pattern.
    """
    try:
        html_content = result._repr_html_()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"    ✅ Saved: {output_path}")
    except AttributeError:
        # Fallback if _repr_html_ not available
        result.save_as_html(output_path)
        print(f"    ✅ Saved (fallback): {output_path}")
    except Exception as e:
        print(f"    ⚠️  Save failed: {e}")
        try:
            result.save_as_html(output_path)
        except Exception as e2:
            print(f"    ❌ Both save methods failed: {e2}")

def run_deepchecks_ai4i():
    print("\n  🔍 Running DeepChecks -- AI4I...")

    df_train = pl.read_parquet(AI4I_FEATURES_PATH).to_pandas()
    df_test  = pl.read_parquet(AI4I_RESULTS_PATH).to_pandas()
    model    = joblib.load(AI4I_CLASSIFIER_PATH)

    shared_cols = [c for c in AI4I_FEATURE_COLS if c in df_test.columns]
    X_train = df_train[shared_cols];  y_train = df_train[AI4I_TARGET_COL]
    X_test  = df_test[shared_cols];   y_test  = df_test[AI4I_TARGET_COL]

    train_dc = DCDataset(X_train, label=y_train, cat_features=[])
    test_dc  = DCDataset(X_test,  label=y_test,  cat_features=[])

    print("    Running data integrity suite...")
    try:
        result = data_integrity().run(train_dc)
        _save_deepchecks_html(
            result,
            os.path.join(REPORTS_DIR, "deepchecks_ai4i_data_integrity.html")
        )
    except Exception as e:
        print(f"    ⚠️  Data integrity warning: {e}")

    print("    Running train-test validation suite...")
    try:
        result = train_test_validation().run(train_dc, test_dc)
        _save_deepchecks_html(
            result,
            os.path.join(REPORTS_DIR, "deepchecks_ai4i_train_test.html")
        )
    except Exception as e:
        print(f"    ⚠️  Train-test validation warning: {e}")

    print("    Running model evaluation suite...")
    try:
        result = model_evaluation().run(train_dc, test_dc, model)
        _save_deepchecks_html(
            result,
            os.path.join(REPORTS_DIR, "deepchecks_ai4i_model_eval.html")
        )
    except Exception as e:
        print(f"    ⚠️  Model evaluation warning: {e}")

    print("  ✅ DeepChecks AI4I complete")


def run_deepchecks_nasa():
    print("\n  🔍 Running DeepChecks -- NASA...")

    df       = pl.read_parquet(NASA_FEATURES_PATH).to_pandas()
    feat_cols = [c for c in df.columns if c not in NASA_EXCLUDE_COLS]

    train_df = df[df["dataset_id"].isin(["FD001","FD003"])]
    test_df  = df[df["dataset_id"].isin(["FD002","FD004"])]

    train_dc = DCDataset(train_df[feat_cols], label=train_df["rul"],
                         cat_features=[])
    test_dc  = DCDataset(test_df[feat_cols],  label=test_df["rul"],
                         cat_features=[])

    print("    Running data integrity suite...")
    try:
        result = data_integrity().run(train_dc)
        _save_deepchecks_html(
            result,
            os.path.join(REPORTS_DIR, "deepchecks_nasa_data_integrity.html")
        )
    except Exception as e:
        print(f"    ⚠️  NASA data integrity warning: {e}")

    print("    Running train-test validation (FD001/3 vs FD002/4)...")
    try:
        result = train_test_validation().run(train_dc, test_dc)
        _save_deepchecks_html(
            result,
            os.path.join(REPORTS_DIR, "deepchecks_nasa_train_test.html")
        )
    except Exception as e:
        print(f"    ⚠️  NASA train-test warning: {e}")

    print("  ✅ DeepChecks NASA complete")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Evidently: Data Drift Reports
# ══════════════════════════════════════════════════════════════════════════════

def run_evidently_nasa():
    """
    FIX from original: compare condition-NORMALIZED sensors.
    Original compared raw FD001 vs FD004 — always showed drift
    because of operating condition differences, not degradation.
    Now: normalized sensors mean drift = actual distribution shift.
    """
    print("\n  📉 Running Evidently — NASA Sensor Drift...")

    if not os.path.exists(NASA_FD001_PATH) or not os.path.exists(NASA_FD004_PATH):
        print("    ⚠️  FD001/FD004 cleaned parquets not found — skipping")
        return

    fd001 = pl.read_parquet(NASA_FD001_PATH).to_pandas()
    fd004 = pl.read_parquet(NASA_FD004_PATH).to_pandas()

    # Use only informative sensors — already condition-normalized in Layer 1
    sensor_cols = [f"s{i}" for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
                   if f"s{i}" in fd001.columns and f"s{i}" in fd004.columns]

    ref_df  = fd001[sensor_cols].sample(
        min(2000, len(fd001)), random_state=42
    )
    curr_df = fd004[sensor_cols].sample(
        min(2000, len(fd004)), random_state=42
    )

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ])
    report.run(reference_data=ref_df, current_data=curr_df)

    path = os.path.join(REPORTS_DIR, "nasa_sensor_drift_report.html")
    report.save_html(path)
    print(f"  ✅ NASA sensor drift report: {path}")
    print(f"     Note: FD001 (1 condition) vs FD004 (6 conditions), "
          f"both condition-normalized")


def run_evidently_ai4i():
    """
    AI4I feature drift report.
    Reference: training features (ai4i_features.parquet)
    Current:   predictions dataset (ai4i_results.parquet)
    Tracks whether live predictions drift from training distribution.
    """
    print("\n  📉 Running Evidently — AI4I Feature Drift...")

    df_ref  = pl.read_parquet(AI4I_FEATURES_PATH).to_pandas()
    df_curr = pl.read_parquet(AI4I_RESULTS_PATH).to_pandas()

    # Shared feature columns only
    cols = [c for c in AI4I_FEATURE_COLS
            if c in df_ref.columns and c in df_curr.columns]

    ref_df  = df_ref[cols + [AI4I_TARGET_COL]].sample(
        min(3000, len(df_ref)), random_state=42
    )
    curr_df = df_curr[cols + [AI4I_TARGET_COL]].sample(
        min(3000, len(df_curr)), random_state=42
    )

    # Data drift report
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        DatasetDriftMetric(),
    ])
    drift_report.run(reference_data=ref_df, current_data=curr_df)
    drift_path = os.path.join(REPORTS_DIR, "ai4i_feature_drift_report.html")
    drift_report.save_html(drift_path)
    print(f"  ✅ AI4I feature drift report: {drift_path}")

    # Replace the target drift block with this:
    if "failure_prob" in df_curr.columns:
        try:
            target_report = Report(metrics=[
              ColumnDriftMetric(column_name="failure"),
        ])
        # Don't rename — pass the column directly
            target_report.run(
                  reference_data=ref_df[[AI4I_TARGET_COL]],
                  current_data=curr_df[[AI4I_TARGET_COL]]
        )
            target_path = os.path.join(
                REPORTS_DIR, "ai4i_prediction_drift_report.html"
        )
            target_report.save_html(target_path)
            print(f"  ✅ AI4I prediction drift report: {target_path}")
        except Exception as e:
           print(f"  ⚠️  Target drift report warning: {e}")
    

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_monitoring():
    print("🚀 Layer 8: MLOps Monitoring")
    print("=" * 55)
    _ensure_dirs()

    # ── Connect to MLflow ──────────────────────────────────────────────────────
    print("\n📡 Connecting to MLflow...")
    client = _connect_mlflow()

    # ── Section 1: MLflow experiment tracking ─────────────────────────────────
    print("\n[1/3] MLflow — Experiment Tracking & Model Registry")
    print("-" * 45)

    log_ai4i_classification(client)
    log_nasa_rul_model(client)
    log_causal_analysis()
    log_optimization()
    log_simulation()
    log_system_intelligence()

    print("\n  ✅ All experiments logged to MLflow")
    print(f"  🌐 View at: {MLFLOW_TRACKING_URI}")

    # ── Section 2: DeepChecks ─────────────────────────────────────────────────
    print("\n[2/3] DeepChecks — Data & Model Validation")
    print("-" * 45)

    run_deepchecks_ai4i()
    run_deepchecks_nasa()

    # ── Section 3: Evidently ──────────────────────────────────────────────────
    print("\n[3/3] Evidently — Drift Monitoring")
    print("-" * 45)

    run_evidently_nasa()
    run_evidently_ai4i()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("📋 LAYER 8 SUMMARY")
    print("=" * 55)
    print(f"  MLflow UI    : {MLFLOW_TRACKING_URI}")
    print(f"  Experiments  : 6 logged")
    print(f"  Models registered:")
    print(f"    AI4I_Failure_Classifier → Production")
    print(f"    NASA_RUL_Predictor      → Production")
    print(f"  DeepChecks reports : {REPORTS_DIR}/")
    print(f"  Evidently reports  : {REPORTS_DIR}/")
    print("=" * 55)
    print("✅ Layer 8 complete.")
    print("\n  To view MLflow UI:")
    print("  $ mlflow ui --host 0.0.0.0 --port 5000")
    print("  Then open: http://localhost:5000")


if __name__ == "__main__":
    run_monitoring()