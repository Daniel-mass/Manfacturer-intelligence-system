# src/optimizer.py
import json
import os
import math
import numpy as np
import pandas as pd
import polars as pl
import joblib
from scipy.optimize import linprog

# ── paths ──────────────────────────────────────────────────────────────────────
AI4I_CAUSAL_PATH   = "models/ai4i_causal_report.json"
NASA_CAUSAL_PATH   = "models/nasa_causal_report.json"
NASA_MODEL_PATH    = "models/nasa_rul_model.joblib"
NASA_FEATURES_PATH = "data/processed/nasa_features.parquet"
AI4I_OPT_OUT       = "models/ai4i_optimizer_results.json"
NASA_OPT_OUT       = "models/nasa_optimizer_results.json"

# ── constants from dataset documentation ──────────────────────────────────────
OSF_THRESHOLD   = {0: 11000, 1: 12000, 2: 13000}
PROD_TYPE_LABEL = {0: "L", 1: "M", 2: "H"}
PWF_MIN_WATTS   = 3500
PWF_MAX_WATTS   = 9000
TWF_ONSET_MIN   = 200


# ══════════════════════════════════════════════════════════════════════════════
# AI4I — OR-Tools CP-SAT
# ══════════════════════════════════════════════════════════════════════════════
class AI4IOptimizer:
    """
    Constraint optimization for AI4I machine setpoints.

    Problem reframed: given a target power output (operator-defined),
    find the RPM/Torque split that minimizes OSF failure risk while
    hitting that power target.

    At low wear: optimizer pushes torque high (efficient, lower RPM)
    At high wear: OSF forces torque down, RPM must compensate
    → genuinely different setpoints per tool wear scenario

    All constraints from dataset documentation and Layer 4 causal report.
    """

    def __init__(self):
        from ortools.sat.python import cp_model
        self.cp_model = cp_model
        self._load_causal_context()

    def _load_causal_context(self):
        with open(AI4I_CAUSAL_PATH) as f:
            causal = json.load(f)
        tw = causal.get("tool_wear_effect", {})
        self.danger_threshold      = tw.get("danger_threshold_minutes", 200)
        self.tw_effect_per_10min   = tw.get("effect_per_10_minutes_pct", 0.305)
        pw = causal.get("power_effect", {})
        self.power_effect_per_100w = pw.get("effect_per_100_watts_pct", 0.300)
        print(f"  Causal context loaded:")
        print(f"    Danger threshold    : {self.danger_threshold} min")
        print(f"    TW effect           : {self.tw_effect_per_10min:.3f}% per 10 min")
        print(f"    Power effect        : {self.power_effect_per_100w:.4f}% per 100W")

    def solve(self, current_tool_wear: int, prod_type: int = 1,
              target_power_w: float = 7000.0) -> dict:
        """
        Given a target power output, find RPM/Torque that:
        - Hits the target power (±5% tolerance)
        - Maximizes torque within constraints
          (higher torque = lower RPM for same power = less mechanical stress)
        - Stays inside all failure mode constraints

        Why this produces different setpoints per tool wear:
        - New tool (0 min): OSF allows torque up to 75Nm → optimizer
          pushes torque high, RPM drops to stay at target power
        - Worn tool (180 min): OSF caps torque at 48Nm → optimizer
          forced to lower torque, RPM rises to compensate
        - End of life (235 min): OSF caps torque at 37Nm → even lower
          torque, even higher RPM
        Same power target, completely different setpoints.
        """
        model  = self.cp_model.CpModel()
        rpm    = model.NewIntVar(1200, 2800, "rpm")
        torque = model.NewIntVar(10, 75, "torque")
        osf_limit = OSF_THRESHOLD[prod_type]

        # ── Constraint 1: OSF ─────────────────────────────────────────────────
        safe_torque_osf = int((osf_limit * 0.8) / max(current_tool_wear, 1))
        safe_torque_osf = max(10, min(75, safe_torque_osf))
        model.Add(torque <= safe_torque_osf)
        print(f"    OSF  : torque ≤ {safe_torque_osf} Nm "
              f"(type={PROD_TYPE_LABEL[prod_type]}, limit={osf_limit})")

        # ── Constraint 2: TWF ─────────────────────────────────────────────────
        if current_tool_wear >= TWF_ONSET_MIN:
            model.Add(torque <= 40)
            print(f"    TWF  : wear={current_tool_wear} ≥ {TWF_ONSET_MIN}, "
                  f"torque ≤ 40 Nm")

        # ── Constraint 3: Causal danger threshold ────────────────────────────
        if current_tool_wear >= self.danger_threshold:
            model.Add(torque <= 35)
            print(f"    Causal: wear={current_tool_wear} ≥ "
                  f"{self.danger_threshold:.0f}, torque ≤ 35 Nm")

        # ── Constraint 4: PWF ─────────────────────────────────────────────────
        power_product = model.NewIntVar(0, 75 * 2800, "power_product")
        model.AddMultiplicationEquality(power_product, [torque, rpm])
        model.Add(power_product * 105 >= PWF_MIN_WATTS * 1000)
        model.Add(power_product * 105 <= PWF_MAX_WATTS * 1000)
        print(f"    PWF  : power in [{PWF_MIN_WATTS}, {PWF_MAX_WATTS}] W")

        # ── Constraint 5: Hit target power ±5% ───────────────────────────────
        # power_product × 105 / 1000 ≈ actual_power_watts
        # So: power_product ≈ target_power_w × 1000 / 105
        target_product = int(target_power_w * 1000 / 105)
        tolerance      = int(target_product * 0.05)
        model.Add(power_product >= target_product - tolerance)
        model.Add(power_product <= target_product + tolerance)
        print(f"    Target: {target_power_w:.0f}W ±5% "
              f"(product range [{target_product-tolerance}, "
              f"{target_product+tolerance}])")

        # ── Objective: Maximize torque ────────────────────────────────────────
        # At same power target, higher torque = lower RPM.
        # Lower RPM reduces mechanical fatigue and bearing stress.
        # OSF constraint limits torque as wear increases →
        # optimizer is forced to a lower torque + higher RPM setpoint
        # for worn tools vs new tools. Different results per scenario.
        model.Maximize(torque)

        solver = self.cp_model.CpSolver()
        status = solver.Solve(model)

        if status in [self.cp_model.OPTIMAL, self.cp_model.FEASIBLE]:
            res_rpm      = solver.Value(rpm)
            res_torque   = solver.Value(torque)
            actual_power = res_torque * (res_rpm * 2 * math.pi / 60)
            osf_load     = current_tool_wear * res_torque
            osf_margin   = (1 - osf_load / osf_limit) * 100 if osf_limit > 0 else 100

            power_fail_risk = (actual_power / 100) * self.power_effect_per_100w
            tw_fail_risk    = (current_tool_wear / 10) * self.tw_effect_per_10min

            if current_tool_wear >= self.danger_threshold:
                urgency = "CRITICAL"
            elif current_tool_wear >= TWF_ONSET_MIN:
                urgency = "WARNING"
            elif current_tool_wear >= 150:
                urgency = "CAUTION"
            else:
                urgency = "NORMAL"

            result = {
                "status":               "OPTIMAL" if status == self.cp_model.OPTIMAL else "FEASIBLE",
                "urgency":              urgency,
                "prod_type":            PROD_TYPE_LABEL[prod_type],
                "current_tool_wear_min": current_tool_wear,
                "target_power_w":       target_power_w,
                "optimal_rpm":          res_rpm,
                "optimal_torque_nm":    res_torque,
                "actual_power_w":       round(actual_power, 2),
                "power_in_safe_range":  PWF_MIN_WATTS <= actual_power <= PWF_MAX_WATTS,
                "osf_load":             round(osf_load, 1),
                "osf_threshold":        osf_limit,
                "osf_margin_pct":       round(osf_margin, 1),
                "failure_risk": {
                    "from_tool_wear_pct":      round(tw_fail_risk, 3),
                    "from_power_pct":          round(power_fail_risk, 4),
                    "combined_additional_pct": round(tw_fail_risk + power_fail_risk, 3)
                },
                "constraints_applied": {
                    "osf":    f"torque ≤ {safe_torque_osf} Nm",
                    "twf":    "active (torque ≤ 40)" if current_tool_wear >= TWF_ONSET_MIN else "inactive",
                    "causal": "active (torque ≤ 35)" if current_tool_wear >= self.danger_threshold else "inactive",
                    "pwf":    f"power in [{PWF_MIN_WATTS}, {PWF_MAX_WATTS}] W"
                },
                "dashboard_recommendation": (
                    f"Optimal for {target_power_w:.0f}W target: "
                    f"RPM={res_rpm}, Torque={res_torque}Nm, "
                    f"Actual power={actual_power:.0f}W. "
                    f"OSF margin: {osf_margin:.1f}%. "
                    f"Urgency: {urgency}."
                )
            }
            print(f"    ✅ RPM={res_rpm}, Torque={res_torque}Nm, "
                  f"Power={actual_power:.0f}W, OSF margin={osf_margin:.1f}%, "
                  f"Urgency={urgency}")
            return result

        else:
            print(f"    ❌ Infeasible at {target_power_w:.0f}W target — "
                  f"wear={current_tool_wear}, type={PROD_TYPE_LABEL[prod_type]}")
            # Retry at lower power target
            if target_power_w > PWF_MIN_WATTS + 500:
                print(f"    Retrying at {target_power_w - 1000:.0f}W...")
                return self.solve(current_tool_wear, prod_type,
                                  target_power_w - 1000)
            return {
                "status":   "INFEASIBLE",
                "urgency":  "CRITICAL",
                "prod_type": PROD_TYPE_LABEL[prod_type],
                "current_tool_wear_min": current_tool_wear,
                "target_power_w": target_power_w,
                "optimal_rpm": 0,
                "optimal_torque_nm": 0,
                "actual_power_w": 0,
                "osf_margin_pct": 0,
                "dashboard_recommendation":
                    "No safe setpoint — schedule immediate tool replacement"
            }

    def run_scenarios(self) -> dict:
        print("\n🔧 AI4I Optimizer: Running scenarios...")

        # FIX: target power decreases as tool wear increases
        # New tool can run at high power, worn tool runs conservatively
        scenarios = [
            {"tool_wear": 0,   "label": "New tool",             "target_power": 8000},
            {"tool_wear": 100, "label": "Mid-life tool",         "target_power": 7500},
            {"tool_wear": 180, "label": "Approaching TWF onset", "target_power": 7000},
            {"tool_wear": 210, "label": "Past TWF onset",        "target_power": 6000},
            {"tool_wear": 235, "label": "Near end of life",      "target_power": 5000},
        ]

        results = {}
        for prod_type in [0, 1, 2]:
            pt_label = PROD_TYPE_LABEL[prod_type]
            results[pt_label] = {}
            print(f"\n  Product Type {pt_label} (OSF={OSF_THRESHOLD[prod_type]}):")
            for s in scenarios:
                print(f"\n  [{s['label']} — {s['tool_wear']} min, "
                      f"target={s['target_power']}W]:")
                results[pt_label][s["label"]] = self.solve(
                    s["tool_wear"], prod_type, s["target_power"]
                )

        os.makedirs("models", exist_ok=True)
        with open(AI4I_OPT_OUT, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n✅ AI4I results saved: {AI4I_OPT_OUT}")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# NASA — Linear Programming on Causal Estimates
# ══════════════════════════════════════════════════════════════════════════════
class NASAOptimizer:
    """
    Intervention optimizer for NASA engine RUL extension.

    Objective: minimize thermal stress within urgency-scaled bounds.
    Intervention aggressiveness scales with RUL urgency:
    - CRITICAL (<20):  40% max reduction
    - WARNING (20-50): 30% max reduction
    - CAUTION (50-80): 20% max reduction
    - NORMAL  (>80):   no intervention

    Both causal estimate and model prediction reported.
    Dashboard shows both as lower/upper bound of expected gain.
    """

    def __init__(self):
        self.model = joblib.load(NASA_MODEL_PATH)
        self._load_causal_context()

    def _load_causal_context(self):
        with open(NASA_CAUSAL_PATH) as f:
            causal = json.load(f)
        ts = causal.get("thermal_stress_effect", {})
        self.thermal_coef    = ts.get("coefficient_per_1std", -8.6275)
        self.causal_reliable = ts.get("reliable", False)
        print(f"  Causal context loaded:")
        print(f"    Thermal coef     : {self.thermal_coef:.4f} cycles/std")
        print(f"    Reliable         : {self.causal_reliable}")

    def get_urgency_and_bound(self, current_rul: float) -> tuple:
        if current_rul < 20:
            return "CRITICAL", 0.40
        elif current_rul < 50:
            return "WARNING",  0.30
        elif current_rul < 80:
            return "CAUTION",  0.20
        else:
            return "NORMAL",   0.0

    def get_model_rul_after_intervention(
        self, feature_row: pd.Series, feature_cols: list,
        optimal_stress: float, current_stress: float
    ) -> float:
        """
        Adjust s11, s15 and thermal_stress proportionally, then
        ask LightGBM for the post-intervention RUL prediction.
        thermal_stress = s11 × s15 → scale each by sqrt(ratio).
        """
        adjusted_row = feature_row.copy()
        if abs(current_stress) > 1e-6:
            scale = optimal_stress / current_stress
            if "s11" in adjusted_row.index:
                adjusted_row["s11"] = adjusted_row["s11"] * math.sqrt(abs(scale))
            if "s15" in adjusted_row.index:
                adjusted_row["s15"] = adjusted_row["s15"] * math.sqrt(abs(scale))
            if "thermal_stress" in adjusted_row.index:
                adjusted_row["thermal_stress"] = optimal_stress

        X = pd.DataFrame(
            [adjusted_row[feature_cols].values],
            columns=feature_cols
        )
        return float(self.model.predict(X)[0])

    def compute_intervention(
        self, feature_row: pd.Series, feature_cols: list,
        current_thermal_stress: float, current_rul: float
    ) -> dict:

        if not self.causal_reliable:
            return {
                "status": "CAUSAL_UNRELIABLE",
                "recommendation": "Causal estimate failed refutation — no intervention"
            }

        urgency, max_reduction = self.get_urgency_and_bound(current_rul)

        if max_reduction == 0.0 or current_thermal_stress <= 0.1:
            model_rul = self.get_model_rul_after_intervention(
                feature_row, feature_cols,
                current_thermal_stress, current_thermal_stress
            )
            return {
                "status":               "NO_INTERVENTION_NEEDED",
                "urgency":              urgency,
                "current_thermal_stress": round(current_thermal_stress, 4),
                "current_rul_cycles":   round(current_rul, 1),
                "model_predicted_rul":  round(model_rul, 1),
                "expected_rul_gain_cycles": 0.0,
                "dashboard_recommendation": (
                    f"Engine healthy. RUL: {current_rul:.1f} cycles "
                    f"(model: {model_rul:.1f}). No intervention needed."
                )
            }

        lower_bound = current_thermal_stress * (1 - max_reduction)
        upper_bound = current_thermal_stress

        res = linprog([1.0], bounds=[(lower_bound, upper_bound)], method="highs")

        if res.success:
            optimal_stress   = float(res.x[0])
            stress_reduction = current_thermal_stress - optimal_stress
            causal_rul_gain  = abs(self.thermal_coef) * stress_reduction
            model_rul_after  = self.get_model_rul_after_intervention(
                feature_row, feature_cols, optimal_stress, current_thermal_stress
            )
            model_rul_gain = max(0.0, model_rul_after - current_rul)

            # Honest labeling: causal is upper bound (linear assumption),
            # model is conservative estimate (nonlinear, near-failure uncertainty)
            return {
                "status":                   "SUCCESS",
                "urgency":                  urgency,
                "current_thermal_stress":   round(current_thermal_stress, 4),
                "optimal_thermal_stress":   round(optimal_stress, 4),
                "stress_reduction_std":     round(stress_reduction, 4),
                "stress_reduction_pct":     round(
                    (stress_reduction / abs(current_thermal_stress)) * 100, 1
                ),
                "max_reduction_allowed_pct": round(max_reduction * 100, 0),
                "current_rul_cycles":        round(current_rul, 1),
                "causal_rul_gain_cycles":    round(causal_rul_gain, 1),
                "causal_note":               "Linear upper bound — assumes constant rate of improvement",
                "model_rul_after_intervention": round(model_rul_after, 1),
                "model_rul_gain_cycles":     round(model_rul_gain, 1),
                "model_note":                "Conservative estimate — LightGBM nonlinear, high uncertainty near failure",
                "causal_coefficient_used":   round(self.thermal_coef, 4),
                "interpretation": (
                    f"Reducing thermal stress by {stress_reduction:.2f} std "
                    f"({stress_reduction / abs(current_thermal_stress) * 100:.1f}%, "
                    f"urgency-scaled max={max_reduction*100:.0f}%). "
                    f"Causal upper bound: +{causal_rul_gain:.1f} cycles. "
                    f"Model conservative estimate: "
                    f"{current_rul:.1f} → {model_rul_after:.1f} cycles."
                ),
                "dashboard_recommendation": (
                    f"[{urgency}] Reduce thermal stress by "
                    f"{stress_reduction / abs(current_thermal_stress) * 100:.0f}%. "
                    f"Expected RUL gain: +{model_rul_gain:.1f} to +{causal_rul_gain:.1f} cycles. "
                    f"RUL range after intervention: "
                    f"{current_rul + model_rul_gain:.1f}–{current_rul + causal_rul_gain:.1f} cycles."
                )
            }
        else:
            return {"status": "INFEASIBLE", "message": res.message}

    def run_scenarios(self) -> dict:
        print("\n🚀 NASA Optimizer: Running intervention scenarios...")

        df = pl.read_parquet(NASA_FEATURES_PATH).to_pandas()
        exclude = ["unit", "cycle", "dataset_id", "condition_cluster", "rul"]
        feature_cols = [c for c in df.columns if c not in exclude]

        scenarios = {
            "critical":  df[df["rul"] < 20].sample(1, random_state=42),
            "degrading": df[(df["rul"] >= 20) & (df["rul"] < 50)].sample(1, random_state=42),
            "healthy":   df[df["rul"] >= 80].sample(1, random_state=42),
        }

        results = {}
        for label, sample in scenarios.items():
            if sample.empty:
                continue
            row         = sample.iloc[0]
            current_rul = float(self.model.predict(
                pd.DataFrame([row[feature_cols].values], columns=feature_cols)
            )[0])
            current_ts  = float(row["thermal_stress"])

            print(f"\n  Scenario: {label}")
            print(f"    Model RUL       : {current_rul:.1f} cycles")
            print(f"    Thermal stress  : {current_ts:.4f} std")

            intervention = self.compute_intervention(
                row, feature_cols, current_ts, current_rul
            )
            results[label] = intervention

            if intervention["status"] == "SUCCESS":
                print(f"    Causal gain     : +{intervention['causal_rul_gain_cycles']:.1f} cycles (upper bound)")
                print(f"    Model gain      : +{intervention['model_rul_gain_cycles']:.1f} cycles (conservative)")
                print(f"    {intervention['interpretation']}")
            else:
                print(f"    Status          : {intervention['status']}")
                if "dashboard_recommendation" in intervention:
                    print(f"    {intervention['dashboard_recommendation']}")

        os.makedirs("models", exist_ok=True)
        with open(NASA_OPT_OUT, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n✅ NASA results saved: {NASA_OPT_OUT}")
        return results


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Layer 5: Optimization")
    print("=" * 50)

    print("\n[1/2] AI4I Constraint Optimizer (OR-Tools CP-SAT)")
    ai4i_opt     = AI4IOptimizer()
    ai4i_results = ai4i_opt.run_scenarios()

    print("\n[2/2] NASA Intervention Optimizer (Linear Programming)")
    nasa_opt     = NASAOptimizer()
    nasa_results = nasa_opt.run_scenarios()

    print("\n" + "=" * 50)
    print("📋 LAYER 5 SUMMARY")
    print("=" * 50)

    print("\n  AI4I Setpoints vs Tool Wear (Product M):")
    for label, result in ai4i_results.get("M", {}).items():
        rpm  = result.get("optimal_rpm", 0)
        trq  = result.get("optimal_torque_nm", 0)
        pwr  = result.get("actual_power_w", 0)
        tgt  = result.get("target_power_w", 0)
        osf  = result.get("osf_margin_pct", 0)
        urg  = result.get("urgency", "")
        print(f"    {label:30s}: RPM={rpm}, Torque={trq}Nm, "
              f"Power={pwr:.0f}W (tgt={tgt:.0f}W), "
              f"OSF margin={osf:.1f}%, [{urg}]")

    print()
    for label, result in nasa_results.items():
        print(f"  NASA {label:10s}: "
              f"{result.get('dashboard_recommendation', result.get('status'))}")

    print("=" * 50)
    print("✅ Layer 5 complete. Results saved to models/")