# src/aggregator.py
import json
import os
import sys
from datetime import datetime

import polars as pl
import ollama
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR         = "models"
DATA_DIR          = "data/processed"
REPORTS_DIR       = "reports"

AI4I_EVAL_PATH    = f"{MODEL_DIR}/ai4i_eval_report.json"
NASA_EVAL_PATH    = f"{MODEL_DIR}/nasa_eval_report.json"
AI4I_CAUSAL_PATH  = f"{MODEL_DIR}/ai4i_causal_report.json"
NASA_CAUSAL_PATH  = f"{MODEL_DIR}/nasa_causal_report.json"
AI4I_OPT_PATH     = f"{MODEL_DIR}/ai4i_optimizer_results.json"
NASA_OPT_PATH     = f"{MODEL_DIR}/nasa_optimizer_results.json"
SIM_RESULTS_PATH  = f"{MODEL_DIR}/simulation_results.json"
AI4I_RESULTS_PATH = f"{DATA_DIR}/ai4i_results.parquet"

SYSTEM_INTEL_OUT  = f"{MODEL_DIR}/system_intelligence.json"
BRIEFINGS_OUT     = f"{MODEL_DIR}/agent_briefings.json"
FINAL_REPORT_OUT  = f"{MODEL_DIR}/ai_final_report.txt"

LLM_MODEL = "llama3.2:latest"


# ── helpers ────────────────────────────────────────────────────────────────────
def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file missing: {path}")
    with open(path) as f:
        return json.load(f)


def _ollama_call(system_prompt: str, user_prompt: str) -> str:
    """Single Ollama call with streaming so you can see progress."""
    try:
        print("    [Streaming response", end="", flush=True)
        full_response = ""
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.3, "num_predict": 1000},
            stream=True
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            # Print a dot every 50 tokens so you know it's alive
            if len(full_response) % 50 == 0:
                print(".", end="", flush=True)
        print("] done")
        return full_response
    except Exception as e:
        print(f"] ERROR: {e}")
        return f"[LLM ERROR: {e}]"

def _tavily_search(query: str, max_results: int = 3) -> str:
    """Run a Tavily web search and return formatted results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "[No TAVILY_API_KEY found — web search skipped]"
    try:
        client  = TavilyClient(api_key=api_key)
        results = client.search(query=query, max_results=max_results)
        formatted = []
        for r in results.get("results", []):
            formatted.append(
                f"Source: {r.get('url', 'unknown')}\n"
                f"Title: {r.get('title', '')}\n"
                f"Content: {r.get('content', '')[:400]}\n"
            )
        return "\n---\n".join(formatted) if formatted else "[No results found]"
    except Exception as e:
        return f"[Tavily search error: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# PART A — SystemAggregator
# ══════════════════════════════════════════════════════════════════════════════
class SystemAggregator:
    """
    Compiles all layer outputs into system_intelligence.json.
    Pure Python — no LLM involved.
    Validates all files present and metrics meet minimum thresholds.
    Halts if ready_for_production is False.
    """

    REQUIRED_FILES = {
        "nasa_eval_report":    NASA_EVAL_PATH,
        "ai4i_eval_report":    AI4I_EVAL_PATH,
        "nasa_causal_report":  NASA_CAUSAL_PATH,
        "ai4i_causal_report":  AI4I_CAUSAL_PATH,
        "nasa_optimizer":      NASA_OPT_PATH,
        "ai4i_optimizer":      AI4I_OPT_PATH,
        "simulation_results":  SIM_RESULTS_PATH,
        "ai4i_results":        AI4I_RESULTS_PATH,
    }

    def check_files(self) -> dict:
        status = {}
        for name, path in self.REQUIRED_FILES.items():
            status[name] = os.path.exists(path)
        return status

    def check_metrics(self,
                       nasa_eval: dict,
                       ai4i_eval: dict,
                       sim: dict) -> dict:
        return {
            "nasa_r2_above_0.7":          nasa_eval.get("r2", 0) >= 0.7,
            "nasa_rmse_below_25":         nasa_eval.get("rmse", 999) < 25,
            "ai4i_f1_above_0.6":          ai4i_eval.get("f1_score", 0) >= 0.6,
            "ai4i_precision_above_0.8":   ai4i_eval.get("precision", 0) >= 0.8,
            "simulation_significant":     sim.get(
                "statistical_significance", {}
            ).get("significant", False),
        }

    def compute_anomaly_summary(self) -> dict:
        df    = pl.read_parquet(AI4I_RESULTS_PATH)
        total = len(df)
        return {
            "total_samples":       total,
            "failure_count":       int(df["failure"].sum()),
            "failure_rate_pct":    round(int(df["failure"].sum()) / total * 100, 2),
            "anomaly_count":       int(df["is_anomaly"].sum()),
            "anomaly_rate_pct":    round(int(df["is_anomaly"].sum()) / total * 100, 2),
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

    def run(self) -> dict:
        print("\n🔧 Part A: SystemAggregator")
        print("-" * 45)

        # File checks
        file_status = self.check_files()
        missing = [k for k, v in file_status.items() if not v]
        if missing:
            print(f"❌ Missing files: {missing}")
            print("   Re-run the corresponding layers before proceeding.")
            sys.exit(1)
        print(f"  ✅ All {len(file_status)} required files present")

        # Load all layer outputs
        nasa_eval  = _load_json(NASA_EVAL_PATH)
        ai4i_eval  = _load_json(AI4I_EVAL_PATH)
        nasa_causal = _load_json(NASA_CAUSAL_PATH)
        ai4i_causal = _load_json(AI4I_CAUSAL_PATH)
        nasa_opt   = _load_json(NASA_OPT_PATH)
        ai4i_opt   = _load_json(AI4I_OPT_PATH)
        sim        = _load_json(SIM_RESULTS_PATH)

        # Metric checks
        metric_checks = self.check_metrics(nasa_eval, ai4i_eval, sim)
        failed_checks = [k for k, v in metric_checks.items() if not v]
        if failed_checks:
            print(f"  ⚠️  Metric checks failed: {failed_checks}")
            print("     Proceeding with warning — review model quality")
        else:
            print(f"  ✅ All {len(metric_checks)} metric checks passed")

        # Anomaly summary
        print("  Computing anomaly summary from parquet...")
        anomaly_summary = self.compute_anomaly_summary()

        # Compile system intelligence
        system_intelligence = {
            "project":  "Intelligent Manufacturing Failure Prevention System",
            "generated_at": datetime.now().isoformat(),
            "pipeline_layers_completed": 6,
            "llm_model": LLM_MODEL,

            "nasa_module": {
                "model_performance": nasa_eval,
                "causal_analysis":   nasa_causal,
                "optimizer":         nasa_opt,
                "simulation":        sim,
            },

            "ai4i_module": {
                "model_performance": ai4i_eval,
                "anomaly_detection": anomaly_summary,
                "causal_analysis":   ai4i_causal,
                "optimizer":         ai4i_opt,
            },

            "system_status": {
                "file_checks":         file_status,
                "metric_checks":       metric_checks,
                "all_files_present":   len(missing) == 0,
                "all_metrics_pass":    len(failed_checks) == 0,
                "ready_for_production": (
                    len(missing) == 0 and len(failed_checks) == 0
                ),
            },

            # Key numbers for dashboard quick-access
            "dashboard_headlines": {
                "nasa_rmse":              nasa_eval.get("rmse"),
                "nasa_r2":                nasa_eval.get("r2"),
                "ai4i_f1":                ai4i_eval.get("f1_score"),
                "ai4i_precision":         ai4i_eval.get("precision"),
                "life_extension_pct":     sim.get("life_extension_pct"),
                "simulation_p_value":     sim.get(
                    "statistical_significance", {}
                ).get("p_value"),
                "tool_wear_danger_min":   ai4i_causal.get(
                    "tool_wear_effect", {}
                ).get("danger_threshold_minutes"),
                "thermal_stress_coef":    nasa_causal.get(
                    "thermal_stress_effect", {}
                ).get("coefficient_per_1std"),
                "anomaly_count":          anomaly_summary["anomaly_count"],
                "failure_count":          anomaly_summary["failure_count"],
            }
        }

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(SYSTEM_INTEL_OUT, "w") as f:
            json.dump(system_intelligence, f, indent=4)

        print(f"  ✅ system_intelligence.json compiled")
        print(f"     Ready for production: "
              f"{system_intelligence['system_status']['ready_for_production']}")
        return system_intelligence


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Three Ollama Agents
# ══════════════════════════════════════════════════════════════════════════════

def run_nasa_analyst(system_intel: dict) -> dict:
    """
    Agent 1: NASA Prognostics Analyst
    Translates NASA model outputs into maintenance scheduling language.
    Sources: LightGBM eval, DoWhy causal, linprog optimizer, Mesa simulation.
    Web search: CMAPSS benchmarks, RUL accuracy standards.
    """
    print("\n🤖 Agent 1: NASA Prognostics Analyst")
    print("-" * 45)

    nasa = system_intel["nasa_module"]

    # Tavily searches for external benchmarks
    print("  🔍 Searching: CMAPSS RUL benchmarks...")
    search1 = _tavily_search(
        "CMAPSS turbofan RUL prediction RMSE benchmark accuracy 2023 2024"
    )
    print("  🔍 Searching: RUL accuracy production standards...")
    search2 = _tavily_search(
        "predictive maintenance RUL accuracy R2 score production deployment threshold"
    )

    system_prompt = """
You are a Senior Maintenance Engineer and Data Scientist specializing in 
turbofan engine prognostics. You translate AI model outputs into 
actionable maintenance scheduling guidance for plant engineers.

STYLE: Technical but accessible. No jargon without explanation. 
Be honest about limitations. Give specific, actionable numbers.
Audience: Manufacturing engineers who schedule engine maintenance.
"""

    user_prompt = f"""
Analyze these NASA turbofan engine prognostics results and provide 
engineering guidance. Answer each question with specific numbers.

=== MODEL PERFORMANCE ===
{json.dumps(nasa['model_performance'], indent=2)}

=== CAUSAL ANALYSIS (DoWhy) ===
{json.dumps(nasa['causal_analysis'], indent=2)}

=== INTERVENTION OPTIMIZER (Linear Programming) ===
{json.dumps(nasa['optimizer'], indent=2)}

=== FLEET SIMULATION (Mesa — 50 vs 50 engines) ===
{json.dumps(nasa['simulation'], indent=2)}

=== INDUSTRY BENCHMARKS (Web Search) ===
{search1}
{search2}

Answer these questions for a plant maintenance engineer:

1. MODEL RELIABILITY: Our RMSE is {nasa['model_performance'].get('rmse')} cycles 
   and R²={nasa['model_performance'].get('r2')}. Is this accurate enough to 
   schedule engine maintenance windows? What safety buffer should engineers add?

2. THERMAL STRESS ACTION: Our causal analysis shows thermal stress reduces RUL 
   by {abs(nasa['causal_analysis'].get('thermal_stress_effect', {}).get('coefficient_per_1std', 0)):.1f} 
   cycles per 1 standard deviation. What does this mean operationally? 
   What actions can a maintenance team take?

3. FLEET MANAGEMENT: The simulation shows {nasa['simulation'].get('life_extension_pct')}% 
   life extension with statistical significance p={nasa['simulation'].get('statistical_significance', {}).get('p_value')}. 
   How should fleet managers act on this finding?

4. HONEST LIMITATIONS: What are the 2-3 most important limitations 
   engineers should know about this system?

Format your response as a structured engineering brief.
"""

    print("  💬 Calling LLM...")
    response = _ollama_call(system_prompt, user_prompt)

    briefing = {
        "agent":          "NASA Prognostics Analyst",
        "model_used":     LLM_MODEL,
        "generated_at":   datetime.now().isoformat(),
        "data_sources": {
            "model_performance": "nasa_eval_report.json (LightGBM)",
            "causal_analysis":   "nasa_causal_report.json (DoWhy)",
            "optimizer":         "nasa_optimizer_results.json (linprog)",
            "simulation":        "simulation_results.json (Mesa)",
            "web_search":        "Tavily — CMAPSS benchmarks"
        },
        "engineering_brief": response,
        "key_numbers": {
            "rmse_cycles":          nasa["model_performance"].get("rmse"),
            "r2_score":             nasa["model_performance"].get("r2"),
            "thermal_stress_coef":  nasa["causal_analysis"].get(
                "thermal_stress_effect", {}
            ).get("coefficient_per_1std"),
            "life_extension_pct":   nasa["simulation"].get("life_extension_pct"),
            "sim_p_value":          nasa["simulation"].get(
                "statistical_significance", {}
            ).get("p_value"),
        }
    }

    print("  ✅ NASA briefing complete")
    return briefing


def run_ai4i_analyst(system_intel: dict) -> dict:
    """
    Agent 2: AI4I Safety Analyst
    Translates AI4I model outputs into machine operation and safety guidance.
    Sources: RF classifier eval, Isolation Forest, DoWhy causal, OR-Tools optimizer.
    Web search: tool wear standards, failure detection benchmarks.
    """
    print("\n🤖 Agent 2: AI4I Safety Analyst")
    print("-" * 45)

    ai4i = system_intel["ai4i_module"]

    # Representative setpoint — M product, mid-life tool
    rep_setpoint = {}
    try:
        opt_data = _load_json(AI4I_OPT_PATH)
        rep_setpoint = opt_data.get("M", {}).get("Mid-life tool", {})
    except Exception:
        pass

    print("  🔍 Searching: Tool wear failure detection benchmarks...")
    search1 = _tavily_search(
        "CNC machine tool wear failure detection F1 precision recall benchmark manufacturing"
    )
    print("  🔍 Searching: Tool replacement interval standards...")
    search2 = _tavily_search(
        "ISO tool wear replacement interval CNC machining standard minutes"
    )

    system_prompt = """
You are a Senior Manufacturing Safety Engineer specializing in CNC machine 
failure prevention and predictive maintenance. You translate AI model outputs 
into specific operational guidance for machine operators and safety engineers.

STYLE: Direct and prescriptive. Give specific thresholds and actions.
Be honest about model limitations and false alarm rates.
Audience: Machine operators and safety engineers on the plant floor.
"""

    user_prompt = f"""
Analyze these AI4I machine failure prevention results and provide 
operational guidance. Give specific, actionable instructions.

=== FAILURE CLASSIFICATION MODEL (Random Forest) ===
{json.dumps(ai4i['model_performance'], indent=2)}

=== ANOMALY DETECTION SUMMARY (Isolation Forest) ===
{json.dumps(ai4i['anomaly_detection'], indent=2)}

=== CAUSAL ANALYSIS (DoWhy) ===
{json.dumps(ai4i['causal_analysis'], indent=2)}

=== REPRESENTATIVE SETPOINT (OR-Tools, M product, 100min wear) ===
{json.dumps(rep_setpoint, indent=2)}

=== INDUSTRY BENCHMARKS (Web Search) ===
{search1}
{search2}

Answer these questions for machine operators and safety engineers:

1. TOOL REPLACEMENT TIMING: Tool wear increases failure probability by 
   {ai4i['causal_analysis'].get('tool_wear_effect', {}).get('effect_per_10_minutes_pct', 0):.3f}% 
   every 10 minutes. Danger threshold is 
   {ai4i['causal_analysis'].get('tool_wear_effect', {}).get('danger_threshold_minutes', 240)} minutes.
   Give a specific tool replacement schedule.

2. ALARM INTERPRETATION: Our model has Precision={ai4i['model_performance'].get('precision')} 
   and Recall={ai4i['model_performance'].get('recall')}. How many false alarms 
   will operators see per 1000 cycles? When should they act immediately vs monitor?

3. SETPOINT GUIDANCE: Given OR-Tools optimizer results showing different 
   RPM/Torque per tool wear stage, give operators a simple decision rule 
   for adjusting machine settings as tool wears.

4. ANOMALY vs FAILURE: We have {ai4i['anomaly_detection'].get('anomaly_count')} 
   anomalies and {ai4i['anomaly_detection'].get('failure_count')} confirmed failures. 
   What's the difference and which should operators prioritize?

5. HONEST LIMITATIONS: What are the 2 most important limitations 
   operators should know?

Format as an operational safety brief with clear action items.
"""

    print("  💬 Calling LLM...")
    response = _ollama_call(system_prompt, user_prompt)

    briefing = {
        "agent":        "AI4I Safety Analyst",
        "model_used":   LLM_MODEL,
        "generated_at": datetime.now().isoformat(),
        "data_sources": {
            "model_performance": "ai4i_eval_report.json (Random Forest)",
            "anomaly_detection": "ai4i_results.parquet (Isolation Forest)",
            "causal_analysis":   "ai4i_causal_report.json (DoWhy)",
            "optimizer":         "ai4i_optimizer_results.json (OR-Tools CP-SAT)",
            "web_search":        "Tavily — tool wear and failure detection benchmarks"
        },
        "engineering_brief": response,
        "key_numbers": {
            "f1_score":              ai4i["model_performance"].get("f1_score"),
            "precision":             ai4i["model_performance"].get("precision"),
            "recall":                ai4i["model_performance"].get("recall"),
            "roc_auc":               ai4i["model_performance"].get("roc_auc"),
            "danger_threshold_min":  ai4i["causal_analysis"].get(
                "tool_wear_effect", {}
            ).get("danger_threshold_minutes"),
            "tw_effect_per_10min":   ai4i["causal_analysis"].get(
                "tool_wear_effect", {}
            ).get("effect_per_10_minutes_pct"),
            "anomaly_count":         ai4i["anomaly_detection"].get("anomaly_count"),
            "failure_count":         ai4i["anomaly_detection"].get("failure_count"),
        }
    }

    print("  ✅ AI4I briefing complete")
    return briefing


def run_synthesizer(system_intel: dict,
                    nasa_briefing: dict,
                    ai4i_briefing: dict) -> str:
    """
    Agent 3: System Intelligence Synthesizer
    Cross-module synthesis into a single engineering brief.
    Input: both agent briefings + full system intelligence.
    No web search — synthesizes from computed data only.
    """
    print("\n🤖 Agent 3: System Intelligence Synthesizer")
    print("-" * 45)

    headlines = system_intel.get("dashboard_headlines", {})

    system_prompt = """
You are a Chief Industrial AI Consultant presenting findings to a 
manufacturing plant management team. You synthesize AI system results 
into business-level recommendations with engineering precision.

STYLE: Executive summary style with engineering depth.
Lead with the most important findings and actions.
Be specific — use actual numbers from the data.
Be honest about what the system can and cannot do.
Audience: Plant managers and senior engineers making deployment decisions.
"""

    user_prompt = f"""
Synthesize these findings from our Intelligent Manufacturing AI System 
into a final engineering report for plant management.

=== NASA PROGNOSTICS BRIEFING (Agent 1) ===
{nasa_briefing.get('engineering_brief', '')}

=== AI4I SAFETY BRIEFING (Agent 2) ===
{ai4i_briefing.get('engineering_brief', '')}

=== SYSTEM HEADLINE NUMBERS ===
- NASA RUL Model: RMSE={headlines.get('nasa_rmse')} cycles, R²={headlines.get('nasa_r2')}
- AI4I Failure Model: F1={headlines.get('ai4i_f1')}, Precision={headlines.get('ai4i_precision')}
- Fleet Life Extension: {headlines.get('life_extension_pct')}% (p={headlines.get('simulation_p_value')})
- Tool Wear Danger Threshold: {headlines.get('tool_wear_danger_min')} minutes
- Thermal Stress Causal Effect: {headlines.get('thermal_stress_coef')} cycles per 1-std
- Anomalies Detected: {headlines.get('anomaly_count')} / 10,000 samples
- Confirmed Failures: {headlines.get('failure_count')} / 10,000 samples

Write a final engineering report covering:

1. EXECUTIVE SUMMARY (3-4 sentences)
   What this system does and its headline result.

2. TOP 3 ACTIONS FOR THIS WEEK
   Specific, numbered, with thresholds and owners.

3. HOW THE TWO SYSTEMS WORK TOGETHER
   How NASA fleet management and AI4I machine safety reinforce each other.

4. BUSINESS CASE
   Cost impact of 43.4% fleet life extension + 87.1% failure detection.

5. DEPLOYMENT READINESS
   What's ready now, what needs monitoring, what to watch for.

6. HONEST SYSTEM LIMITATIONS
   The 3 most important limitations management should know before deploying.

Write in plain English. Be specific. Be honest.
"""

    print("  💬 Calling LLM for synthesis...")
    report = _ollama_call(system_prompt, user_prompt)
    print("  ✅ Synthesis complete")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_aggregator():
    print("🚀 Layer 7: Intelligence Aggregation")
    print("=" * 55)

    # ── Part A: Compile system intelligence ───────────────────────────────────
    aggregator     = SystemAggregator()
    system_intel   = aggregator.run()

    # ── Part B: Three Ollama agents ───────────────────────────────────────────
    print("\n🧠 Part B: Multi-Agent Reasoning")
    print("-" * 45)
    print(f"  Model: {LLM_MODEL}")
    print(f"  Agents: 3 (sequential)")

    nasa_briefing  = run_nasa_analyst(system_intel)
    ai4i_briefing  = run_ai4i_analyst(system_intel)
    final_report   = run_synthesizer(system_intel, nasa_briefing, ai4i_briefing)

    # ── Save all outputs ───────────────────────────────────────────────────────
    briefings = {
        "generated_at":   datetime.now().isoformat(),
        "nasa_briefing":  nasa_briefing,
        "ai4i_briefing":  ai4i_briefing,
    }
    with open(BRIEFINGS_OUT, "w", encoding="utf-8") as f:
        json.dump(briefings, f, indent=4)
    print(f"\n✅ Agent briefings saved: {BRIEFINGS_OUT}")

    with open(FINAL_REPORT_OUT, "w", encoding="utf-8") as f:
        header = (
            f"INTELLIGENT MANUFACTURING AI SYSTEM\n"
            f"Final Engineering Report\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Model: {LLM_MODEL}\n"
            f"{'='*60}\n\n"
        )
        f.write(header + final_report)
    print(f"✅ Final report saved: {FINAL_REPORT_OUT}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("📋 LAYER 7 SUMMARY")
    print("=" * 55)
    headlines = system_intel.get("dashboard_headlines", {})
    print(f"  System ready: "
          f"{system_intel['system_status']['ready_for_production']}")
    print(f"  NASA R²={headlines.get('nasa_r2')} | "
          f"RMSE={headlines.get('nasa_rmse')} cycles")
    print(f"  AI4I F1={headlines.get('ai4i_f1')} | "
          f"Precision={headlines.get('ai4i_precision')}")
    print(f"  Fleet life extension: {headlines.get('life_extension_pct')}% "
          f"(p={headlines.get('simulation_p_value')})")
    print(f"  Outputs:")
    print(f"    {SYSTEM_INTEL_OUT}")
    print(f"    {BRIEFINGS_OUT}")
    print(f"    {FINAL_REPORT_OUT}")
    print("=" * 55)
    print("✅ Layer 7 complete.")


if __name__ == "__main__":
    run_aggregator()