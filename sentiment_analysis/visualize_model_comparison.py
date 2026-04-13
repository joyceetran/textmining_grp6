"""
visualize_model_comparison.py

Two publication-ready charts from disagreement_analysis.xlsx:
  1. Overall predicted-label distribution per model (grouped bar)
  2. Per-gold-class prediction breakdown per model (stacked bar / confusion-as-bars)

Run from repo root:
    python sentiment_analysis/visualize_model_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

DATA_PATH = "sentiment_analysis/disagreement_analysis.xlsx"
OUT_DIR   = "sentiment_analysis"

LABELS     = ["positive", "neutral", "negative"]
LABEL_COLORS = {
    "positive": "#4CAF50",   # green
    "neutral":  "#9E9E9E",   # grey
    "negative": "#F44336",   # red
}

MODEL_COLS   = ["finbert", "naive_bayes", "rule_based"]
MODEL_LABELS = ["FinBERT", "Naive Bayes", "Rule-Based (LM)"]

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH)
n  = len(df)
print(f"Loaded {n:,} rows  |  gold distribution: {df['gold'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 1 – Overall predicted-label distribution per model
# ══════════════════════════════════════════════════════════════════════════════

# Build a dict: model → {label: count}
dist = {
    col: df[col].value_counts().reindex(LABELS, fill_value=0)
    for col in MODEL_COLS
}
# Also add gold for reference
dist["gold"] = df["gold"].value_counts().reindex(LABELS, fill_value=0)

all_models  = ["gold"]   + MODEL_COLS
all_labels_display = ["Gold\n(ground truth)"] + MODEL_LABELS

x     = np.arange(len(all_models))
width = 0.22
offsets = [-1, 0, 1]  # three label groups

fig, ax = plt.subplots(figsize=(10, 5.5))

for i, (label, color) in enumerate(LABEL_COLORS.items()):
    counts = [dist[col][label] for col in all_models]
    pcts   = [c / n * 100 for c in counts]
    bars   = ax.bar(x + offsets[i] * width, pcts, width, label=label.capitalize(),
                    color=color, edgecolor="white", linewidth=0.6)
    # value labels on bars
    for bar, pct in zip(bars, pcts):
        if pct >= 1.5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(all_labels_display, fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel("% of test sample", fontsize=11)
ax.set_title("Predicted label distribution per model\n(stratified test sample, 500 sentences per gold class)",
             fontsize=12, pad=12)
ax.legend(title="Predicted label", fontsize=9, title_fontsize=9)
ax.set_ylim(0, 75)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
out1 = f"{OUT_DIR}/chart1_overall_distribution.png"
plt.savefig(out1, dpi=150)
plt.close()
print(f"Saved → {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 2 – Per-gold-class breakdown: stacked bars per model
# For each gold label → what % does each model predict as pos/neu/neg?
# ══════════════════════════════════════════════════════════════════════════════

gold_groups = LABELS  # positive, neutral, negative

fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
fig.suptitle(
    "How each model distributes predictions — broken down by gold label\n"
    "(stacked bars show predicted pos / neu / neg for each gold class)",
    fontsize=12, y=1.01
)

for ax, gold_lbl in zip(axes, gold_groups):
    subset = df[df["gold"] == gold_lbl]
    n_sub  = len(subset)

    # model order on x-axis
    x_pos  = np.arange(len(MODEL_COLS))
    bottom = np.zeros(len(MODEL_COLS))

    for pred_lbl, color in LABEL_COLORS.items():
        heights = []
        for col in MODEL_COLS:
            cnt = (subset[col] == pred_lbl).sum()
            heights.append(cnt / n_sub * 100)

        bars = ax.bar(x_pos, heights, bottom=bottom,
                      color=color, label=pred_lbl.capitalize(),
                      edgecolor="white", linewidth=0.5)

        # label segments ≥ 5%
        for bar, h, b in zip(bars, heights, bottom):
            if h >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        b + h / 2,
                        f"{h:.0f}%",
                        ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottom += np.array(heights)

    ax.set_title(f"Gold: {gold_lbl.upper()}\n(n = {n_sub:,})", fontsize=11, pad=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODEL_LABELS, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

axes[0].set_ylabel("% of sentences in that gold class", fontsize=10)

# single shared legend
handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, title="Predicted label",
           loc="lower center", ncol=3, fontsize=9, title_fontsize=9,
           bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
out2 = f"{OUT_DIR}/chart2_per_goldclass_breakdown.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# Chart 3 – Summary: per-sentiment-class comparison of all models vs ground truth
# Grouped bars: x-axis = class, bars within group = Gold / FinBERT / NB / RB
# ══════════════════════════════════════════════════════════════════════════════

ALL_COLS   = ["gold"] + MODEL_COLS
ALL_LABELS = ["Gold\n(ground truth)"] + MODEL_LABELS
MODEL_COLORS = ["#37474F", "#1976D2", "#7B1FA2", "#F57C00"]  # dark-grey, blue, purple, orange

x3      = np.arange(len(LABELS))   # 3 groups: positive, neutral, negative
n_mod   = len(ALL_COLS)
w3      = 0.17
offsets = np.linspace(-(n_mod - 1) / 2, (n_mod - 1) / 2, n_mod)

fig, ax = plt.subplots(figsize=(11, 5.5))

for i, (col, label, color) in enumerate(zip(ALL_COLS, ALL_LABELS, MODEL_COLORS)):
    pcts = [dist[col][lbl] / n * 100 for lbl in LABELS]
    bars = ax.bar(x3 + offsets[i] * w3, pcts, w3,
                  label=label.replace("\n", " "),
                  color=color, edgecolor="white", linewidth=0.6)
    for bar, pct in zip(bars, pcts):
        if pct >= 2:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{pct:.0f}%",
                    ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x3)
ax.set_xticklabels([lbl.capitalize() for lbl in LABELS], fontsize=12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel("% of test sample", fontsize=11)
ax.set_title(
    "Predicted sentiment distribution — all models vs ground truth\n"
    "(grouped by class; each bar = % of full test sample labelled that class)",
    fontsize=12, pad=12,
)
ax.legend(title="Model", fontsize=9, title_fontsize=9, loc="upper right")
ax.set_ylim(0, 85)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
out3 = f"{OUT_DIR}/chart3_summary_by_class.png"
plt.savefig(out3, dpi=150)
plt.close()
print(f"Saved → {out3}")


# ── Quick numeric summary to console ──────────────────────────────────────────
print("\n── Chart 1 numeric summary (% of full sample) ──────────────────────────────")
header = f"  {'Label':<10}" + "".join(f"  {m:>14}" for m in ["Gold"] + MODEL_LABELS)
print(header)
for lbl in LABELS:
    vals = "".join(
        f"  {dist[col][lbl]/n*100:>13.1f}%"
        for col in ["gold"] + MODEL_COLS
    )
    print(f"  {lbl:<10}{vals}")

print("\n── Chart 2 numeric summary (% within each gold class) ──────────────────────")
for gold_lbl in gold_groups:
    sub = df[df["gold"] == gold_lbl]
    print(f"\n  Gold = {gold_lbl.upper()} (n={len(sub)})")
    print(f"  {'Model':<20}" + "".join(f"  {p:>10}" for p in ["Pred POS", "Pred NEU", "Pred NEG"]))
    for col, lbl in zip(MODEL_COLS, MODEL_LABELS):
        row = "".join(
            f"  {(sub[col]==p).sum()/len(sub)*100:>9.1f}%"
            for p in ["positive", "neutral", "negative"]
        )
        print(f"  {lbl:<20}{row}")

print("\nDone.")
