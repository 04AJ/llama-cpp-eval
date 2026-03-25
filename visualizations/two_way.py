import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    os.path.join(os.path.dirname(__file__),
                 "../evaluation/numa_compare_llama-8b_default/numa_compare_7829121.csv")
)

df = pd.read_csv(CSV)
df["n_threads"] = df["n_threads"].astype(int)

pp = df[df["n_prompt"] > 0].copy()   # prompt processing (512 tokens)
tg = df[df["n_gen"]    > 0].copy()   # token generation  (128 tokens)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("llama 8B Q8_0 — original vs mbind fix (no --numa)", fontsize=13)

COLORS = {"original": "#888888", "fixed": "#2196F3"}
LABELS = {"original": "original (no NUMA)", "fixed": "fixed (LLAMA_NUMA_BIND_ROWS=1)"}

for ax, data, title, ylabel in [
    (axes[0], pp, "Prompt processing  (pp, 512 tokens)", "tokens / sec"),
    (axes[1], tg, "Token generation   (tg, 128 tokens)", "tokens / sec"),
]:
    for binary, grp in data.groupby("binary"):
        grp = grp.sort_values("n_threads")
        ax.plot(grp["n_threads"], grp["avg_ts"],
                marker="o", label=LABELS[binary],
                color=COLORS.get(binary, None), linewidth=2)

    # mark the socket boundary
    ax.axvline(x=40, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(40.5, ax.get_ylim()[1] * 0.02, "socket\nboundary",
            color="red", fontsize=8, va="bottom")

    ax.set_title(title)
    ax.set_xlabel("threads")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(data["n_threads"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = CSV.replace(".csv", "_two_way.png")
plt.savefig(out, dpi=150)
print(f"saved: {out}")
plt.show()

# ---------------------------------------------------------------------------
# Table: original vs fixed vs delta for both tests
# ---------------------------------------------------------------------------
def make_table_fig(data, label):
    orig  = data[data["binary"] == "original"].sort_values("n_threads")
    fixed = data[data["binary"] == "fixed"].sort_values("n_threads")
    deltas     = (fixed["avg_ts"].values - orig["avg_ts"].values).round(2)
    delta_pcts = (deltas / orig["avg_ts"].values * 100).round(1)

    rows = list(zip(
        orig["n_threads"].astype(int),
        orig["avg_ts"].round(2),
        fixed["avg_ts"].round(2),
        deltas,
        delta_pcts,
    ))
    col_labels = ["threads", "original tok/s", "fixed tok/s", "delta tok/s", "delta %"]

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

    # colour header
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#444444")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    # colour delta % cells: green = improvement, red = regression
    for row_idx, (_, _, _, _, dpct) in enumerate(rows, start=1):
        color = "#c8e6c9" if dpct > 0 else "#ffcdd2" if dpct < 0 else "#ffffff"
        tbl[row_idx, 4].set_facecolor(color)

    plt.tight_layout()
    return fig

fig_pp = make_table_fig(pp, "Prompt processing (pp, 512 tokens)")
fig_tg = make_table_fig(tg, "Token generation  (tg, 128 tokens)")

out_pp = CSV.replace(".csv", "_table_pp.png")
out_tg = CSV.replace(".csv", "_table_tg.png")
fig_pp.savefig(out_pp, dpi=150, bbox_inches="tight")
fig_tg.savefig(out_tg, dpi=150, bbox_inches="tight")
print(f"saved: {out_pp}")
print(f"saved: {out_tg}")
fig_pp.show()
fig_tg.show()
