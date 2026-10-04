"""Step 6 - Static figures for the README (outputs/figures/*.png)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

BLUE, RED, GRAY, INK, INK2, GRID = "#2a78d6", "#e34948", "#a3abb6", "#111418", "#4a5260", "#e3e6eb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def T(name):
    return pd.read_csv(C.TABLES / f"{name}.csv")


def pct_axis(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_product():
    d = T("02_product_relief").sort_values("relief_rate_pct")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(d))
    ax.barh(y, d["relief_rate_pct"] / 100, height=0.55, color=BLUE)
    ax.set_yticks(y, [f"{p}  ({n:,})" for p, n in zip(d["product"], d["complaints"])])
    for yi, v in zip(y, d["relief_rate_pct"]):
        ax.text(v / 100 + 0.004, yi, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    pct_axis(ax)
    ax.set_xlim(0, 0.34)
    ax.set_title("Share of complaints closed with relief, by product (complaint count)")
    fig.savefig(C.FIGURES / "relief_by_product.png")
    plt.close(fig)


def fig_themes():
    d = T("theme_summary")
    d = d[d["theme"] != "Too short to classify"].copy()
    d["gap"] = d["relief_rate"] - d["expected_relief_given_products"]
    d = d.sort_values("gap")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    y = np.arange(len(d))
    for yi, e, a in zip(y, d["expected_relief_given_products"], d["relief_rate"]):
        ax.plot([e, a], [yi, yi], color=BLUE if a >= e else RED, linewidth=2, solid_capstyle="round")
    ax.scatter(d["expected_relief_given_products"], y, s=46, facecolor="white", edgecolor=INK2,
               linewidth=1.5, zorder=3, label="Expected from its product mix")
    ax.scatter(d["relief_rate"], y, s=52, color=[BLUE if g >= 0 else RED for g in d["gap"]], zorder=4,
               edgecolor="white", linewidth=1.5, label="Actual relief rate")
    ax.set_yticks(y, [f"{t}  ({n:,})" for t, n in zip(d["theme"], d["complaints"])])
    pct_axis(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("AI-discovered complaint themes: actual vs. expected relief rate")
    fig.savefig(C.FIGURES / "theme_relief.png")
    plt.close(fig)


def fig_language():
    d = T("05_language_signals")[["signal", "mix_adjusted_gap_pts", "complaints_with_signal"]]
    tone = T("sentiment_vs_relief")
    extra = []
    for q, label in [(0, "Most negative tone (bottom 20%)"), (4, "Least negative tone (top 20%)")]:
        r = tone.iloc[q]
        extra.append({"signal": label,
                      "mix_adjusted_gap_pts": round(100 * (r["relief_rate"] - r["expected_relief_given_products"]), 1),
                      "complaints_with_signal": int(r["complaints"])})
    d = pd.concat([d, pd.DataFrame(extra)]).sort_values("mix_adjusted_gap_pts")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(d))
    colors = [BLUE if v >= 0 else RED for v in d["mix_adjusted_gap_pts"]]
    ax.barh(y, d["mix_adjusted_gap_pts"], height=0.55, color=colors)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, d["signal"])
    for yi, v in zip(y, d["mix_adjusted_gap_pts"]):
        ax.text(v + (0.25 if v >= 0 else -0.25), yi, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.set_xlim(-6, 12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Relief rate vs. expected for the same product mix (percentage points)")
    ax.set_title("What consumers write about matters more than how they write it")
    fig.savefig(C.FIGURES / "language_signals.png")
    plt.close(fig)


def fig_models():
    cmp_ = T("relief_model_comparison")
    dec = T("relief_deciles")
    auto = T("routing_automation").dropna()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    ax = axes[0]
    names = ["Structured\nfields only", "Narrative\ntext only", "Combined"]
    vals = cmp_["roc_auc"].values
    ax.bar(range(3), vals, width=0.5, color=[GRAY, GRAY, BLUE])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3), names)
    ax.set_ylim(0.5, 0.86)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Relief model: ROC AUC (0.5 = random)")

    ax = axes[1]
    x = np.r_[0, dec["decile"].values / 10]
    yv = np.r_[0, dec["cumulative_capture"].values]
    ax.plot(x, yv, color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.5, label="Random order")
    top20 = dec.loc[dec["decile"] == 2, "cumulative_capture"].iloc[0]
    ax.annotate(f"Top 20% of queue\ncaptures {top20:.0%} of relief", (0.2, top20), (0.5, 0.16),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Complaints reviewed (highest score first)")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Cumulative gains (combined model)")

    ax = axes[2]
    ax.plot(auto["coverage"], auto["accuracy_on_auto_routed"], color=BLUE, linewidth=2)
    ok = auto[auto["accuracy_on_auto_routed"] >= 0.95].iloc[0]
    ax.scatter([ok["coverage"]], [ok["accuracy_on_auto_routed"]], s=50, color=BLUE, edgecolor="white", zorder=3)
    ax.annotate(f"{ok['coverage']:.0%} auto-routed\nat {ok['accuracy_on_auto_routed']:.0%} accuracy",
                (ok["coverage"], ok["accuracy_on_auto_routed"]), (0.42, 0.9), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of complaints routed automatically")
    ax.set_ylabel("Routing accuracy")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Routing model: automation trade-off")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(C.FIGURES / "model_performance.png")
    plt.close(fig)


def main():
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig_product()
    fig_themes()
    fig_language()
    fig_models()
    print(f"Figures -> {C.FIGURES.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    main()
