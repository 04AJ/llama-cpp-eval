import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

CSV = sys.argv[1] if len(sys.argv) > 1 else (
    os.path.join(os.path.dirname(__file__),
                 "../evaluation/reproduce_issue/reproduce_issue_7831543.csv")
)

df = pd.read_csv(CSV)
df["n_threads"] = df["n_threads"].astype(int)

pp = df[df["n_prompt"] > 0].sort_values("n_threads")
tg = df[df["n_gen"]    > 0].sort_values("n_threads")

SOCKET_BOUNDARY = 40   # Xeon Gold 6230: 20 cores/socket × 2 sockets

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("llama 8B Q8_0 — performance vs thread count", fontsize=13)

for ax, data, title, ylabel in [
    (axes[0], pp, "Prompt processing  (pp, 512 tokens)", "tokens / sec"),
    (axes[1], tg, "Token generation   (tg, 128 tokens)", "tokens / sec"),
]:
    threads = data["n_threads"].values
    ts      = data["avg_ts"].values

    # shade cross-socket region
    ax.axvspan(SOCKET_BOUNDARY, threads.max() + 4, color="#ffebee", alpha=0.6,
               label="cross-socket region")

    # socket boundary line
    ax.axvline(x=SOCKET_BOUNDARY, color="#e53935", linestyle="--", linewidth=1.5)
    ax.text(SOCKET_BOUNDARY + 0.5, ax.get_ylim()[1],
            "socket boundary\n(40 threads)", color="#e53935",
            fontsize=8, va="top")

    # main line
    ax.plot(threads, ts, marker="o", color="#1565C0", linewidth=2, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("threads")
    ax.set_ylabel(ylabel)
    ax.set_xticks(threads)
    ax.grid(True, alpha=0.3)

    handles = [
        mpatches.Patch(color="#ffebee", label="cross-socket region"),
        plt.Line2D([0], [0], color="#1565C0", marker="o", label="avg tok/s"),
        plt.Line2D([0], [0], color="#e53935", linestyle="--", label="socket boundary"),
    ]
    ax.legend(handles=handles, fontsize=8)

plt.tight_layout()
out = CSV.replace(".csv", "_cliff.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
plt.show()
