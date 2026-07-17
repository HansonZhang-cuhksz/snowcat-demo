"""Figure for the hardware sensitivity-to-fusion sweep (reads result/sensitivity.json).

Run: conda run -n fusion python -m fusion.make_sensitivity_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = Path(__file__).resolve().parent / "result"

# default value per sweep (for the relative-multiplier x-axis)
DEFAULTS = {
    "HBM bandwidth (TB/s)": 2.04e12,
    "Tensor GFLOP/s/core": 512e9,
    "CUDA GFLOP/s/core": 5.64e9,
}
STYLE = {
    "HBM bandwidth (TB/s)": ("tab:blue", "o-"),
    "Tensor GFLOP/s/core": ("tab:orange", "s-"),
    "CUDA GFLOP/s/core": ("tab:green", "^-"),
}


def main() -> None:
    rows = json.load(open(RESULT_DIR / "sensitivity.json"))

    def agg(stage, sweep, metric, reducer):
        vals = sorted({r["param_value"] for r in rows if r["stage"] == stage and r["sweep"] == sweep})
        out = []
        for v in vals:
            g = [r for r in rows if r["stage"] == stage and r["sweep"] == sweep and r["param_value"] == v]
            out.append((v / DEFAULTS[sweep], reducer(abs(r[metric]) for r in g)))
        return zip(*out)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, stage in enumerate(("decode", "prefill")):
        for j, (metric, ylabel, reducer) in enumerate([
            ("dcuda", "max |CUDA-core split shift| (cores)", max),
            ("dtime_pct", "max |time change| (%)", max),
        ]):
            ax = axes[i][j]
            for sweep, (color, ls) in STYLE.items():
                x, y = agg(stage, sweep, metric, reducer)
                ax.plot(x, y, ls, color=color, label=sweep.replace(" GFLOP/s/core", "").replace(" (TB/s)", ""))
            ax.axvline(1.0, color="grey", linestyle=":", linewidth=1)
            ax.set_xlabel("hardware value / default  (1.0 = default setup)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{stage} — {'die-split' if j == 0 else 'time'} sensitivity to fusion")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(title="swept knob (others at default)")

    fig.suptitle("Where is the hardware sensitive to fusion? (max over the 6 fusions; HBM latency omitted — no effect)\n"
                 "Decode split-shift is driven by CUDA-core throughput; prefill time-saving grows at low bandwidth / high tensor throughput.")
    fig.tight_layout()
    out = str(RESULT_DIR / "sensitivity.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
