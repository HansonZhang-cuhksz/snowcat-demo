"""Cross-fusion comparison figure for the PREFILL analyses (verified results).

Run: conda run -n fusion python -m fusion.make_summary_figure_prefill
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, HBM change GiB [+ = increase/hidden, - = saved], time saved ms, CUDA at optimum)
BASELINE_CUDA, BASELINE_TENSOR = 326, 858
ROWS = [
    ("F1 FA+resid", +72.0, 18.95, 326),
    ("F2 FA+resid+RMS", +60.0, 25.96, 326),
    ("F3 RMS+up_gate", -12.0, 7.01, 326),
    ("F4 up_gate+act", -128.0, 74.75, 326),
    ("F5 act+down", +32.0, 74.75, 326),
    ("F6 up_gate+act+down", +384.0, 74.75, 326),
]

RESULT_DIR = Path(__file__).resolve().parent / "result"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r[0] for r in ROWS]
    hbm = [r[1] for r in ROWS]
    tms = [r[2] for r in ROWS]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    colors = ["tab:green" if v < 0 else "tab:red" for v in hbm]
    bars = ax.bar(x, hbm, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("Layer HBM change (GiB)   [<0 = saved, >0 = increase, hidden]")
    ax.set_title("Prefill HBM change per fusion (all hidden — compute-bound)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for b, v in zip(bars, hbm):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    ax = axes[1]
    ax.bar(x, tms, color="tab:blue")
    ax.set_ylabel("Layer time saved (ms)")
    ax.set_title("Prefill time saved per fusion (die split unchanged: 326/858 for all)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for xi, t in zip(x, tms):
        ax.text(xi, t, f"{t:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Six fusions vs unfused baseline — GLM-5.2 PREFILL layer (DSA, compute-bound)\n"
                 "(baseline: 326 CUDA / 858 tensor / 8.086 MiB SMEM, 12992 ms, 434.5 TFLOP/s; "
                 "NO fusion moves the die split)")
    fig.tight_layout()
    out = str(RESULT_DIR / "fusion_summary_prefill.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
