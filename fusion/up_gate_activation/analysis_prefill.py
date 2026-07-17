"""Prefill area-distribution analysis — Fusion 4: up_gate + SwiGLU activation.

Run: conda run -n fusion python -m fusion.up_gate_activation.analysis_prefill
"""

from __future__ import annotations

from pathlib import Path

import prefill_area_latency

from fusion import common
from fusion.up_gate_activation import model

TITLE = "Prefill Fusion 4: up_gate + SwiGLU activation"
RESULT_DIR = Path(__file__).resolve().parent / "result"


def run() -> common.FusionComparison:
    return common.run_fusion(
        model, prefill_area_latency, TITLE, RESULT_DIR,
        "prefill_up_gate_activation_sweep.csv", prefix="prefill_",
    )


if __name__ == "__main__":
    run()
