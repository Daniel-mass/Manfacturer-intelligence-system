# src/causal_analysis.py
import pandas as pd
import polars as pl
import numpy as np
import json
import os
from dowhy import CausalModel

# ── paths ──────────────────────────────────────────────────────────────────────
AI4I_RESULTS_PATH  = "data/processed/ai4i_results.parquet"
AI4I_RAW_PATH      = "data/processed/ai4i_cleaned.parquet"
NASA_FEATURES_PATH = "data/processed/nasa_features.parquet"
AI4I_CAUSAL_OUT    = "models/ai4i_causal_report.json"
NASA_CAUSAL_OUT    = "models/nasa_causal_report.json"


# ── helpers ────────────────────────────────────────────────────────────────────
def get_feature_ranges(raw_parquet_path: str, cols: list) -> dict:
    """
    Read original (unscaled) min/max from Layer 1 output.
    Used to back-transform scaled causal coefficients to real units.
    """
    df = pl.read_parquet(raw_parquet_path).to_pandas()
    return {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in cols if col in df.columns
    }


def run_refutations(model, estimand, estimate, label: str) -> dict:
    """
    Two refutation tests per estimate:
    Placebo:      replace treatment with random noise — effect should vanish
    Random cause: add random confounder — estimate should stay stable
    """
    print(f"    Running refutations for: {label}")
    results = {}
    original_val = float(estimate.value)

    try:
        placebo = model.refute_estimate(
            estimand, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )
        placebo_val = float(placebo.new_effect)
        placebo_passed = abs(placebo_val) < abs(original_val) * 0.2
        results["placebo"] = {
            "original_effect": round(original_val, 6),
            "placebo_effect":  round(placebo_val, 6),
            "passed": placebo_passed
        }
        print(f"      Placebo: original={original_val:.4f}, placebo={placebo_val:.4f}, passed={placebo_passed}")
    except Exception as e:
        results["placebo"] = {"passed": False, "error": str(e)}

    try:
        random_cause = model.refute_estimate(
            estimand, estimate,
            method_name="random_common_cause"
        )
        rc_val = float(random_cause.new_effect)
        rc_passed = abs(rc_val - original_val) < abs(original_val) * 0.2
        results["random_common_cause"] = {
            "original_effect": round(original_val, 6),
            "new_effect":      round(rc_val, 6),
            "passed": rc_passed
        }
        print(f"      Random cause: original={original_val:.4f}, new={rc_val:.4f}, passed={rc_passed}")
    except Exception as e:
        results["random_common_cause"] = {"passed": False, "error": str(e)}

    results["overall_passed"] = all(v.get("passed", False) for v in results.values())
    return results


# ── AI4I causal analysis ───────────────────────────────────────────────────────
def run_ai4i_causal():
    print("\n🔬 AI4I Causal Analysis...")

    df = pl.read_parquet(AI4I_RESULTS_PATH).to_pandas()
    ranges = get_feature_ranges(AI4I_RAW_PATH, ["tool_wear", "power_w", "torque", "rpm"])

    report = {}

    # ── Q1: Tool Wear → Failure ───────────────────────────────────────────────
    print("\n  Q1: Does tool_wear causally increase failure probability?")
    print(f"      Treatment: tool_wear | Outcome: failure")
    print(f"      Confounders: prod_type, torque, rpm")

    try:
        model_tw = CausalModel(
            data=df,
            treatment="tool_wear",
            outcome="failure",
            common_causes=["prod_type", "torque", "rpm"]
        )
        estimand_tw = model_tw.identify_effect(proceed_when_unidentifiable=True)
        estimate_tw = model_tw.estimate_effect(
            estimand_tw,
            method_name="backdoor.linear_regression",
            target_units="ate"
        )

        scaled_coef = float(estimate_tw.value)
        tw_range = ranges["tool_wear"]["max"] - ranges["tool_wear"]["min"]

        # FIX: correct back-transformation
        # scaled_coef is per 1 scaled unit = per tw_range real minutes
        # divide by tw_range → per 1 real minute
        # multiply by 10 → per 10 real minutes
        effect_per_10min = (scaled_coef / tw_range) * 10

        print(f"      Scaled coefficient: {scaled_coef:.6f}")
        print(f"      Tool wear range: {tw_range:.1f} minutes")
        print(f"      Effect per 10 min tool wear: {effect_per_10min*100:.4f}% failure probability change")

        refutations_tw = run_refutations(model_tw, estimand_tw, estimate_tw, "tool_wear→failure")

        # Danger threshold: at what tool wear does failure prob exceed 15%?
        # P(failure) ≈ baseline + scaled_coef * tool_wear_scaled
        # Solve: tool_wear_scaled = (0.15 - baseline) / scaled_coef
        baseline_failure_rate = df["failure"].mean()
        if scaled_coef > 0:
            threshold_scaled = (0.15 - baseline_failure_rate) / scaled_coef
            threshold_scaled = max(0.0, min(1.0, threshold_scaled))
            threshold_minutes = threshold_scaled * tw_range + ranges["tool_wear"]["min"]
            # Hard cap at physical max from dataset description (240 min)
            threshold_minutes = min(threshold_minutes, 240.0)
        else:
            threshold_minutes = 240.0

        print(f"      Danger threshold: {threshold_minutes:.1f} minutes")

        report["tool_wear_effect"] = {
            "scaled_coefficient": round(scaled_coef, 6),
            "tool_wear_range_minutes": round(tw_range, 1),
            "effect_per_10_minutes_pct": round(effect_per_10min * 100, 4),
            "interpretation": (
                f"10 additional minutes of tool wear "
                f"{'increases' if effect_per_10min > 0 else 'decreases'} "
                f"failure probability by {abs(effect_per_10min) * 100:.3f}%"
            ),
            "danger_threshold_minutes": round(float(threshold_minutes), 1),
            "danger_threshold_interpretation": (
                f"Above {threshold_minutes:.0f} min tool wear, "
                f"failure probability exceeds 15%"
            ),
            "refutations": refutations_tw,
            "reliable": refutations_tw["overall_passed"]
        }

    except Exception as e:
        print(f"      ❌ Tool wear causal analysis failed: {e}")
        report["tool_wear_effect"] = {"reliable": False, "error": str(e)}

    # ── Q2: Power → Failure ───────────────────────────────────────────────────
    print("\n  Q2: Does power_w causally affect failure probability?")
    print(f"      Treatment: power_w | Outcome: failure")
    print(f"      Confounders: temp_delta, tool_wear, prod_type")

    try:
        model_pw = CausalModel(
            data=df,
            treatment="power_w",
            outcome="failure",
            common_causes=["temp_delta", "tool_wear", "prod_type"]
        )
        estimand_pw = model_pw.identify_effect(proceed_when_unidentifiable=True)
        estimate_pw = model_pw.estimate_effect(
            estimand_pw,
            method_name="backdoor.linear_regression",
            target_units="ate"
        )

        scaled_coef_pw = float(estimate_pw.value)
        pw_range = ranges["power_w"]["max"] - ranges["power_w"]["min"]

        # FIX: correct back-transformation
        # scaled_coef_pw is per 1 scaled unit = per pw_range real watts
        # divide by pw_range → per 1 real watt
        # multiply by 100 → per 100 real watts
        effect_per_100w = (scaled_coef_pw / pw_range) * 100

        print(f"      Scaled coefficient: {scaled_coef_pw:.6f}")
        print(f"      Power range: {pw_range:.1f} watts")
        print(f"      Effect per 100W: {effect_per_100w*100:.4f}% failure probability change")

        refutations_pw = run_refutations(model_pw, estimand_pw, estimate_pw, "power_w→failure")

        report["power_effect"] = {
            "scaled_coefficient": round(scaled_coef_pw, 6),
            "power_range_watts": round(pw_range, 1),
            "effect_per_100_watts_pct": round(effect_per_100w * 100, 4),
            "interpretation": (
                f"Increasing power by 100W "
                f"{'increases' if effect_per_100w > 0 else 'decreases'} "
                f"failure probability by {abs(effect_per_100w) * 100:.4f}%"
            ),
            "pwf_safe_range_watts": {
                "min": 3500,
                "max": 9000,
                "note": "PWF failure fires outside this range per dataset documentation"
            },
            "refutations": refutations_pw,
            "reliable": refutations_pw["overall_passed"]
        }

    except Exception as e:
        print(f"      ❌ Power causal analysis failed: {e}")
        report["power_effect"] = {"reliable": False, "error": str(e)}

    os.makedirs("models", exist_ok=True)
    with open(AI4I_CAUSAL_OUT, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n✅ AI4I causal report saved: {AI4I_CAUSAL_OUT}")
    return report


# ── NASA causal analysis ───────────────────────────────────────────────────────
def run_nasa_causal():
    print("\n🔬 NASA Causal Analysis...")

    df = pl.read_parquet(NASA_FEATURES_PATH).to_pandas()

    df["dataset_id_enc"] = df["dataset_id"].map(
        {"FD001": 0, "FD002": 1, "FD003": 2, "FD004": 3}
    ).fillna(0).astype(int)

    report = {}

    print("\n  Q1: Does thermal_stress causally reduce RUL?")
    print(f"      Treatment: thermal_stress | Outcome: rul")
    print(f"      Confounders: condition_cluster, dataset_id_enc, cycle")

    # FIX: include_groups=False drops grouping cols from lambda result
    # Re-merge dataset_id_enc and condition_cluster back after sampling
    sample_df = (
        df.groupby(["dataset_id", "condition_cluster"], group_keys=False)
        .apply(
            lambda x: x.sample(min(len(x), 500), random_state=42),
            include_groups=True   # keep group cols in result
        )
        .reset_index(drop=True)
    )
    print(f"      Using stratified sample: {len(sample_df)} rows")
    print(f"      Columns verified: dataset_id_enc present = {'dataset_id_enc' in sample_df.columns}")

    try:
        model_nasa = CausalModel(
            data=sample_df,
            treatment="thermal_stress",
            outcome="rul",
            common_causes=["condition_cluster", "dataset_id_enc", "cycle"]
        )
        estimand_nasa = model_nasa.identify_effect(proceed_when_unidentifiable=True)
        estimate_nasa = model_nasa.estimate_effect(
            estimand_nasa,
            method_name="backdoor.linear_regression",
            target_units="ate"
        )

        # FIX: guard against None estimate value
        if estimate_nasa.value is None:
            raise ValueError(
                "DoWhy returned None estimate — backdoor identification may have failed. "
                "Check that all common_causes columns are present in the dataframe."
            )

        coef_nasa = float(estimate_nasa.value)
        print(f"      Thermal stress coefficient: {coef_nasa:.4f} cycles per 1-std increase")

        refutations_nasa = run_refutations(
            model_nasa, estimand_nasa, estimate_nasa, "thermal_stress→rul"
        )

        report["thermal_stress_effect"] = {
            "coefficient_per_1std": round(coef_nasa, 4),
            "interpretation": (
                f"1 standard deviation increase in thermal stress "
                f"{'reduces' if coef_nasa < 0 else 'increases'} "
                f"RUL by {abs(coef_nasa):.1f} cycles"
            ),
            "unit_note": (
                "thermal_stress is z-score normalized per operating condition. "
                "Coefficient is in RUL cycles per 1 standard deviation."
            ),
            "refutations": refutations_nasa,
            "reliable": refutations_nasa["overall_passed"]
        }

    except Exception as e:
        print(f"      ❌ NASA causal analysis failed: {e}")
        report["thermal_stress_effect"] = {"reliable": False, "error": str(e)}

    with open(NASA_CAUSAL_OUT, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n✅ NASA causal report saved: {NASA_CAUSAL_OUT}")
    return report


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Layer 4: Causal Analysis")
    print("=" * 50)

    ai4i_report = run_ai4i_causal()
    nasa_report  = run_nasa_causal()

    print("\n" + "=" * 50)
    print("📋 CAUSAL ANALYSIS SUMMARY")
    print("=" * 50)

    tw = ai4i_report.get("tool_wear_effect", {})
    pw = ai4i_report.get("power_effect", {})
    ts = nasa_report.get("thermal_stress_effect", {})

    if tw.get("reliable"):
        print(f"✅ Tool Wear:  {tw['interpretation']}")
        print(f"   Danger threshold: {tw['danger_threshold_interpretation']}")
    else:
        print("⚠️  Tool wear effect: inconclusive — refutation failed")

    if pw.get("reliable"):
        print(f"✅ Power:      {pw['interpretation']}")
    else:
        print("⚠️  Power effect: inconclusive — refutation failed")

    if ts.get("reliable"):
        print(f"✅ Thermal Stress: {ts['interpretation']}")
    else:
        print("⚠️  Thermal stress effect: inconclusive — refutation failed")

    print("=" * 50)
    print("✅ Layer 4 complete. Reports saved to models/")