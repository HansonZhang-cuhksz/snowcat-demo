"""Figure for the decode batch-size fusion sweep (reads result/batch_sweep.json).

Run: conda run -n fusion python -m fusion.make_batch_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = Path(__file__).resolve().parent / "result"
FUSIONS = ["F1 FA+resid", "F2 FA+resid+RMS", "F3 RMS+up_gate",
           "F4 up_gate+act", "F5 act+down", "F6 up_gate+act+down"]
COLORS = ["#888", "#aaa", "#5aa", "tab:blue", "tab:red", "tab:green"]


def main() -> None:
    rows = json.load(open(RESULT_DIR / "batch_sweep.json"))
    batches = sorted({r["batch"] for r in rows})

    def series(fusion, field):
        return [next(r[field] for r in rows if r["batch"] == b and r["fusion"] == fusion) for b in batches]

    attn = [next(r["attn_frac"] for r in rows if r["batch"] == b) * 100 for b in batches]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: dtime% vs batch, per fusion.
    ax = axes[0]
    for f, c in zip(FUSIONS, COLORS):
        lw = 2.5 if f.startswith(("F4", "F6")) else 1.3
        ax.plot(batches, series(f, "dtime_pct"), "o-", color=c, linewidth=lw,
                label=f.split(None, 1)[0] + " " + f.split(None, 1)[1])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("decode batch size (tokens)")
    ax.set_ylabel("fusion Δ total time (%)  [>0 = faster]")
    ax.set_title("Fusion time benefit vs batch\n(F6 best at small batch; F4 overtakes at large batch)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel B: attention fraction + Δsplit for F4/F6.
    ax = axes[1]
    ax.plot(batches, attn, "k^-", label="attention time fraction (%)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("decode batch size (tokens)")
    ax.set_ylabel("attention time fraction (%)")
    ax.set_ylim(60, 101)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    for f, c in [("F4 up_gate+act", "tab:blue"), ("F6 up_gate+act+down", "tab:green")]:
        ax2.plot(batches, series(f, "dcuda"), "s--", color=c, label=f"Δsplit {f.split()[0]} (cores)")
    ax2.set_ylabel("die-split shift: unfused−fused CUDA cores")
    ax.set_title("Attention dominates as batch grows;\ndie-split shift (~110–164 cores) is batch-robust")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="center right")

    fig.suptitle("Decode: how batch size affects fusion optimality (default GPU spec, 1M KV context)")
    fig.tight_layout()
    out = str(RESULT_DIR / "batch_sweep.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
