# src/octo_plotter.py
import json
import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────────
AI4I_EVAL_PATH      = "models/ai4i_eval_report.json"
NASA_EVAL_PATH      = "models/nasa_eval_report.json"
AI4I_CAUSAL_PATH    = "models/ai4i_causal_report.json"
AI4I_RESULTS_PATH   = "data/processed/ai4i_results.parquet"
SIM_RESULTS_PATH    = "models/simulation_results.json"
SURVIVAL_CSV_PATH   = "reports/survival_curves.csv"
PLOTS_DIR           = "reports/plots"

# ── style ──────────────────────────────────────────────────────────────────────
MANAGED_COLOR   = "#2196F3"   # blue
UNMANAGED_COLOR = "#F44336"   # red
NORMAL_COLOR    = "#4CAF50"   # green
WARNING_COLOR   = "#FF9800"   # orange
CRITICAL_COLOR  = "#F44336"   # red
ACCENT_COLOR    = "#9C27B0"   # purple

plt.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#e0e0e0",
    "ytick.color":       "#e0e0e0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#333",
    "grid.alpha":        0.4,
    "font.family":       "sans-serif",
    "font.size":         11,
})


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"📁 Plot output directory: {PLOTS_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Survival Curve
# ══════════════════════════════════════════════════════════════════════════════
def plot_survival_curve():
    print("  📈 Generating survival curve...")

    df  = pd.read_csv(SURVIVAL_CSV_PATH)
    sim = json.load(open(SIM_RESULTS_PATH))

    life_ext = sim["life_extension_pct"]
    p_val    = sim["statistical_significance"]["p_value"]
    m_mean   = sim["managed"]["mean_cycles"]
    um_mean  = sim["unmanaged"]["mean_cycles"]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["cycle"], df["survival_pct_managed"],
            color=MANAGED_COLOR, linewidth=2.5, label="AI-Managed Fleet")
    ax.fill_between(df["cycle"], df["survival_pct_managed"],
                    alpha=0.15, color=MANAGED_COLOR)

    ax.plot(df["cycle"], df["survival_pct_unmanaged"],
            color=UNMANAGED_COLOR, linewidth=2.5,
            linestyle="--", label="Unmanaged Fleet")
    ax.fill_between(df["cycle"], df["survival_pct_unmanaged"],
                    alpha=0.10, color=UNMANAGED_COLOR)

    # Annotate divergence
    ax.axvline(x=80, color="#FFD700", linewidth=1,
               linestyle=":", alpha=0.7)
    ax.text(82, 60, "Divergence\npoint",
            color="#FFD700", fontsize=9, alpha=0.9)

    # Mean lines
    ax.axvline(x=m_mean, color=MANAGED_COLOR,
               linewidth=1, linestyle=":", alpha=0.5)
    ax.axvline(x=um_mean, color=UNMANAGED_COLOR,
               linewidth=1, linestyle=":", alpha=0.5)

    # Stats box
    stats_text = (
        f"Life Extension: +{life_ext:.1f}%\n"
        f"Managed mean: {m_mean:.1f} cycles\n"
        f"Unmanaged mean: {um_mean:.1f} cycles\n"
        f"p-value: {p_val:.4f} ✅"
    )
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#0f3460",
                      edgecolor=MANAGED_COLOR,
                      alpha=0.9))

    ax.set_xlabel("Operational Cycles")
    ax.set_ylabel("Fleet Survival (%)")
    ax.set_title("AI-Managed vs Unmanaged Fleet Survival\n"
                 "Mesa Agent-Based Simulation — 50 vs 50 Engines",
                 fontsize=13, pad=15)
    ax.legend(loc="lower left", framealpha=0.3)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "survival_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — NASA Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
def plot_nasa_feature_importance():
    print("  📊 Generating NASA feature importance...")

    report   = json.load(open(NASA_EVAL_PATH))
    features = report.get("top_features", {})

    if not features:
        print("    ⚠️  No top_features in nasa_eval_report — skipping")
        return

    names  = list(features.keys())[:10]
    values = [features[n] for n in names]

    # Normalize to percentage
    total  = sum(values) if sum(values) > 0 else 1
    pcts   = [v / total * 100 for v in values]

    colors = [MANAGED_COLOR if "_trend" in n
              else ACCENT_COLOR if "_velocity" in n
              else "#78909C"
              for n in names]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names[::-1], pcts[::-1],
                   color=colors[::-1], alpha=0.85, height=0.65)

    for bar, pct in zip(bars, pcts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=MANAGED_COLOR,  label="Trend features"),
        mpatches.Patch(color=ACCENT_COLOR,   label="Velocity features"),
        mpatches.Patch(color="#78909C",       label="Raw sensors"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              framealpha=0.3, fontsize=9)

    rmse = report.get("rmse", "N/A")
    r2   = report.get("r2",   "N/A")
    ax.set_title(f"NASA RUL Model — Top Predictive Features\n"
                 f"LightGBM | RMSE={rmse} cycles | R²={r2} (held-out engines)",
                 fontsize=12, pad=15)
    ax.set_xlabel("Relative Importance (%)")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance_nasa.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — AI4I Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
def plot_ai4i_feature_importance():
    print("  📊 Generating AI4I feature importance...")

    report   = json.load(open(AI4I_EVAL_PATH))
    features = report.get("feature_importances", {})

    if not features:
        print("    ⚠️  No feature_importances in ai4i_eval_report — skipping")
        return

    sorted_f = dict(sorted(features.items(),
                            key=lambda x: x[1], reverse=True)[:10])
    names    = list(sorted_f.keys())
    values   = list(sorted_f.values())
    total    = sum(values) if sum(values) > 0 else 1
    pcts     = [v / total * 100 for v in values]

    physics_features  = {"power_w", "stress_index", "overstrain_margin",
                          "temp_delta", "power_w_smooth"}
    sensor_features   = {"rpm", "torque", "tool_wear",
                          "air_temp_c", "proc_temp_c"}

    colors = [NORMAL_COLOR    if n in physics_features
              else WARNING_COLOR if n in sensor_features
              else MANAGED_COLOR
              for n in names]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names[::-1], pcts[::-1],
                   color=colors[::-1], alpha=0.85, height=0.65)

    for bar, pct in zip(bars, pcts[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=NORMAL_COLOR,   label="Physics features"),
        mpatches.Patch(color=WARNING_COLOR,  label="Sensor features"),
        mpatches.Patch(color=MANAGED_COLOR,  label="Smoothed features"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              framealpha=0.3, fontsize=9)

    f1   = report.get("f1_score",  "N/A")
    prec = report.get("precision", "N/A")
    ax.set_title(f"AI4I Failure Model — Top Predictive Features\n"
                 f"Random Forest | F1={f1} | Precision={prec} (held-out test set)",
                 fontsize=12, pad=15)
    ax.set_xlabel("Relative Importance (%)")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance_ai4i.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Failure Mode Breakdown
# ══════════════════════════════════════════════════════════════════════════════
def plot_failure_mode_breakdown():
    print("  🥧 Generating failure mode breakdown...")

    df = pl.read_parquet(AI4I_RESULTS_PATH)

    failure_modes = {
        "TWF\n(Tool Wear)":       int(df["TWF"].sum()),
        "HDF\n(Heat Dissipation)": int(df["HDF"].sum()),
        "PWF\n(Power Failure)":   int(df["PWF"].sum()),
        "OSF\n(Overstrain)":      int(df["OSF"].sum()),
        "RNF\n(Random)":          int(df["RNF"].sum()),
    }

    total_failures = int(df["failure"].sum())
    labels  = list(failure_modes.keys())
    sizes   = list(failure_modes.values())
    colors  = [CRITICAL_COLOR, WARNING_COLOR, MANAGED_COLOR,
               ACCENT_COLOR, "#78909C"]
    explode = [0.05] * len(labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        explode=explode, autopct="%1.1f%%",
        startangle=140, textprops={"color": "#e0e0e0"}
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax1.set_title(f"Failure Mode Distribution\n"
                  f"Total Failures: {total_failures} / 10,000 samples (3.4%)",
                  fontsize=12, pad=15)

    # Bar chart — absolute counts
    bars = ax2.bar(range(len(labels)), sizes,
                   color=colors, alpha=0.85, width=0.6)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Number of Occurrences")
    ax2.set_title("Failure Mode Counts\n(Modes can co-occur)",
                  fontsize=12, pad=15)
    ax2.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 str(val), ha="center", fontsize=10)

    plt.suptitle("AI4I Dataset — Failure Mode Analysis\n"
                 "Source: Ground truth labels from dataset documentation",
                 fontsize=11, y=1.02, color="#b0b0b0")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "failure_mode_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Tool Wear vs Failure Probability
# ══════════════════════════════════════════════════════════════════════════════
def plot_tool_wear_risk():
    print("  ⚠️  Generating tool wear risk scatter...")

    df     = pl.read_parquet(AI4I_RESULTS_PATH).to_pandas()
    causal = json.load(open(AI4I_CAUSAL_PATH))

    danger_threshold = causal.get(
        "tool_wear_effect", {}
    ).get("danger_threshold_minutes", 240)

    tw_effect = causal.get(
        "tool_wear_effect", {}
    ).get("effect_per_10_minutes_pct", 0.305)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: scatter tool_wear vs failure_prob ───────────────────────────────
    normal_mask  = df["failure"] == 0
    failure_mask = df["failure"] == 1

    ax1.scatter(df.loc[normal_mask,  "tool_wear"],
                df.loc[normal_mask,  "failure_prob"],
                alpha=0.3, s=8, color=NORMAL_COLOR,  label="Normal operation")
    ax1.scatter(df.loc[failure_mask, "tool_wear"],
                df.loc[failure_mask, "failure_prob"],
                alpha=0.6, s=15, color=CRITICAL_COLOR, label="Actual failure",
                zorder=5)

    # Danger threshold line
    ax1.axvline(x=danger_threshold,
                color="#FFD700", linewidth=2,
                linestyle="--", label=f"Danger threshold ({danger_threshold:.0f} min)")
    ax1.text(danger_threshold + 2, 0.85,
             f"{danger_threshold:.0f} min\ndanger zone",
             color="#FFD700", fontsize=8)

    # Trend line
    tw_bins   = np.linspace(df["tool_wear"].min(), df["tool_wear"].max(), 30)
    bin_means = []
    for i in range(len(tw_bins) - 1):
        mask = (df["tool_wear"] >= tw_bins[i]) & (df["tool_wear"] < tw_bins[i+1])
        if mask.sum() > 0:
            bin_means.append((tw_bins[i] + tw_bins[i+1]) / 2,)
    bin_centers = [(tw_bins[i] + tw_bins[i+1]) / 2
                   for i in range(len(tw_bins)-1)
                   if ((df["tool_wear"] >= tw_bins[i]) &
                       (df["tool_wear"] < tw_bins[i+1])).sum() > 0]
    bin_probs   = [df.loc[(df["tool_wear"] >= tw_bins[i]) &
                           (df["tool_wear"] < tw_bins[i+1]),
                           "failure_prob"].mean()
                   for i in range(len(tw_bins)-1)
                   if ((df["tool_wear"] >= tw_bins[i]) &
                       (df["tool_wear"] < tw_bins[i+1])).sum() > 0]
    ax1.plot(bin_centers, bin_probs,
             color="#FFD700", linewidth=2, alpha=0.8,
             label="Mean failure prob trend")

    ax1.set_xlabel("Tool Wear (minutes)")
    ax1.set_ylabel("Predicted Failure Probability")
    ax1.set_title("Tool Wear vs Failure Probability\n"
                  "Source: RF Classifier predictions + DoWhy causal threshold",
                  fontsize=11)
    ax1.legend(fontsize=8, framealpha=0.3)
    ax1.grid(True, alpha=0.3)

    # ── Right: risk accumulation bar chart ────────────────────────────────────
    wear_bins   = [0, 50, 100, 150, 200, 240]
    wear_labels = ["0-50", "50-100", "100-150", "150-200", "200-240"]
    bin_failure_rates = []

    for i in range(len(wear_bins) - 1):
        mask = ((df["tool_wear"] >= wear_bins[i]) &
                (df["tool_wear"] < wear_bins[i+1]))
        if mask.sum() > 0:
            rate = df.loc[mask, "failure"].mean() * 100
        else:
            rate = 0
        bin_failure_rates.append(rate)

    bar_colors = [NORMAL_COLOR if r < 2
                  else WARNING_COLOR if r < 8
                  else CRITICAL_COLOR
                  for r in bin_failure_rates]

    bars = ax2.bar(wear_labels, bin_failure_rates,
                   color=bar_colors, alpha=0.85, width=0.6)
    ax2.axhline(y=3.4, color="#FFD700", linewidth=1.5,
                linestyle="--", label="Dataset baseline (3.4%)")

    for bar, rate in zip(bars, bin_failure_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f"{rate:.1f}%", ha="center", fontsize=10)

    causal_text = (
        f"Causal finding:\n"
        f"+{tw_effect:.3f}% failure risk\n"
        f"per 10 min of wear\n"
        f"(DoWhy backdoor adj.)"
    )
    ax2.text(0.97, 0.97, causal_text,
             transform=ax2.transAxes,
             verticalalignment="top",
             horizontalalignment="right",
             fontsize=8,
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#0f3460",
                       edgecolor="#FFD700",
                       alpha=0.9))

    ax2.set_xlabel("Tool Wear Range (minutes)")
    ax2.set_ylabel("Actual Failure Rate (%)")
    ax2.set_title("Failure Rate by Tool Wear Stage\n"
                  "Source: Ground truth labels from ai4i_results.parquet",
                  fontsize=11)
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Tool Wear Risk Analysis — AI4I Manufacturing Dataset",
                 fontsize=12, y=1.02, color="#b0b0b0")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "tool_wear_risk.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✅ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_octo_plotter():
    print("🔭 OctoTools Plotter: Generating dashboard visualizations...")
    print("=" * 55)
    ensure_dirs()

    plots = [
        ("Survival Curve",             plot_survival_curve),
        ("NASA Feature Importance",    plot_nasa_feature_importance),
        ("AI4I Feature Importance",    plot_ai4i_feature_importance),
        ("Failure Mode Breakdown",     plot_failure_mode_breakdown),
        ("Tool Wear Risk",             plot_tool_wear_risk),
    ]

    success = 0
    for name, fn in plots:
        try:
            fn()
            success += 1
        except Exception as e:
            print(f"    ❌ {name} failed: {e}")

    print("=" * 55)
    print(f"✅ OctoTools Plotter complete: {success}/{len(plots)} plots generated")
    print(f"📁 All plots saved to: {PLOTS_DIR}/")

    # Return manifest for aggregator
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "plots_dir":    PLOTS_DIR,
        "plots": {
            "survival_curve":          os.path.join(PLOTS_DIR, "survival_curve.png"),
            "feature_importance_nasa": os.path.join(PLOTS_DIR, "feature_importance_nasa.png"),
            "feature_importance_ai4i": os.path.join(PLOTS_DIR, "feature_importance_ai4i.png"),
            "failure_mode_breakdown":  os.path.join(PLOTS_DIR, "failure_mode_breakdown.png"),
            "tool_wear_risk":          os.path.join(PLOTS_DIR, "tool_wear_risk.png"),
        },
        "success_count": success,
        "total_count":   len(plots)
    }
    return manifest


if __name__ == "__main__":
    manifest = run_octo_plotter()
    print("\nPlot manifest:")
    for k, v in manifest["plots"].items():
        exists = "✅" if os.path.exists(v) else "❌"
        print(f"  {exists} {k}: {v}")