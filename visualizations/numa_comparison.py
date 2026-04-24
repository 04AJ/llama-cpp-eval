import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    os.path.join(os.path.dirname(__file__),
                 "../evaluation/numa_compare_llama-8b_three_way/numa_compare_7831489.csv")
)

df = pd.read_csv(CSV)
df["n_threads"] = df["n_threads"].astype(int)

pp = df[df["n_prompt"] > 0].copy()
tg = df[df["n_gen"]    > 0].copy()

COLORS = {
    "distribute": "#FF9800",
    "fixed":      "#2196F3",
}
LABELS = {
    "distribute": "original (--numa distribute)",
    "fixed":      "fixed (LLAMA_NUMA_BIND_ROWS=1)",
}
ORDER = ["distribute", "fixed"]

# Line plots: distribute vs fixed
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("llama 8B Q8_0 — --numa distribute vs mbind fix", fontsize=13)

for ax, data, title in [
    (axes[0], pp, "Prompt processing  (pp, 512 tokens)"),
    (axes[1], tg, "Token generation   (tg, 128 tokens)"),
]:
    for binary in ORDER:
        grp = data[data["binary"] == binary].sort_values("n_threads")
        if grp.empty:
            continue
        ax.plot(grp["n_threads"], grp["avg_ts"],
                marker="o", label=LABELS[binary],
                color=COLORS[binary], linewidth=2)

    ax.axvline(x=40, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(40.5, ax.get_ylim()[1] * 0.02, "socket\nboundary",
            color="red", fontsize=8, va="bottom")

    ax.set_title(title)
    ax.set_xlabel("threads")
    ax.set_ylabel("tokens / sec")
    ax.set_xticks(sorted(data["n_threads"].unique()))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_lines = CSV.replace(".csv", "_numa_comparison.png")
plt.savefig(out_lines, dpi=150, bbox_inches="tight")
print(f"saved: {out_lines}")
plt.show()

# Tables: distribute / fixed / delta
def make_table_fig(data, label):
    dist = data[data["binary"] == "distribute"].sort_values("n_threads")
    fix  = data[data["binary"] == "fixed"].sort_values("n_threads")

    deltas     = (fix["avg_ts"].values - dist["avg_ts"].values).round(2)
    delta_pcts = (deltas / dist["avg_ts"].values * 100).round(1)

    rows = list(zip(
        dist["n_threads"].astype(int),
        dist["avg_ts"].round(2),
        fix["avg_ts"].round(2),
        deltas,
        [f"{v:+.1f}%" for v in delta_pcts],
    ))
    col_labels = ["threads", "distribute tok/s", "fixed tok/s", "delta tok/s", "delta %"]

    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.4 * len(rows)))
    ax.axis("off")
    ax.set_title(label, fontsize=11, pad=10)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#444444")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    for row_idx, row in enumerate(rows, start=1):
        val = float(row[4].replace("%", ""))
        color = "#c8e6c9" if val > 0 else "#ffcdd2" if val < 0 else "#ffffff"
        tbl[row_idx, 4].set_facecolor(color)

    plt.tight_layout()
    return fig

fig_pp = make_table_fig(pp, "Prompt processing (pp, 512 tokens)")
fig_tg = make_table_fig(tg, "Token generation  (tg, 128 tokens)")

out_pp = CSV.replace(".csv", "_numa_comparison_table_pp.png")
out_tg = CSV.replace(".csv", "_numa_comparison_table_tg.png")
fig_pp.savefig(out_pp, dpi=150, bbox_inches="tight")
fig_tg.savefig(out_tg, dpi=150, bbox_inches="tight")
print(f"saved: {out_pp}")
print(f"saved: {out_tg}")
fig_pp.show()
fig_tg.show()
