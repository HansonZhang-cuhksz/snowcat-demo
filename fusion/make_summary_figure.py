"""Generate the cross-fusion comparison figure from the collected results.

Values are the verified layer-level results of each fusion analysis (see each
fusion's report.md). Run: conda run -n fusion python -m fusion.make_summary_figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, HBM saved MiB, time saved ms, CUDA cores at optimum, tensor cores at optimum)
BASELINE_CUDA, BASELINE_TENSOR = 490, 885
# HBM saved MiB: min-traffic convention (negative = fusion increases HBM, hidden effect).
ROWS = [
    ("F1 FA+resid", -336.0, 0.037, 490, 885),
    ("F2 FA+resid+RMS", -312.0, 0.049, 490, 885),
    ("F3 RMS+up_gate", 24.008, 0.012, 490, 885),
    ("F4 up_gate+act", 256.0, 0.138, 381, 889),
    ("F5 act+down", 128.0, 0.072, 381, 889),
    ("F6 up_gate+act+down", 370.5, 0.204, 354, 890),
]

RESULT_DIR = Path(__file__).resolve().parent / "result"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r[0] for r in ROWS]
    hbm = [r[1] for r in ROWS]
    tms = [r[2] for r in ROWS]
    cuda = [r[3] for r in ROWS]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    bars = ax.bar(x, hbm, color=["tab:blue" if v >= 0 else "tab:red" for v in hbm])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("Layer HBM saved (MiB)  [<0 = increase, hidden]")
    ax.set_title("HBM traffic saved by each fusion (full decode layer, min-traffic conv.)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for b, v, t in zip(bars, hbm, tms):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}\n({t:.3f} ms)",
                ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.axhline(BASELINE_CUDA, color="grey", linestyle="--", label=f"baseline CUDA ({BASELINE_CUDA})")
    ax.plot(x, cuda, "o-", color="tab:red", label="CUDA cores at fused optimum")
    ax.set_ylabel("CUDA cores at optimal area split")
    ax.set_title("Die-partition shift (CUDA cores; lower = more area to tensor/SMEM)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    for xi, c in zip(x, cuda):
        ax.text(xi, c, f"{c}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Six FFN/attention fusions vs unfused baseline — GLM-5.2 decode layer\n"
                 "(baseline: 490 CUDA / 885 tensor / 1.316 MiB SMEM, 1224.492 ms, 2326.16 GiB HBM)")
    fig.tight_layout()
    out = str(RESULT_DIR / "fusion_summary.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
