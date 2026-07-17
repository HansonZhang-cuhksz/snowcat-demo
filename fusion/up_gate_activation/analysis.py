"""Decode area-distribution analysis — Fusion 4: up_gate + SwiGLU activation.

Run: conda run -n fusion python -m fusion.up_gate_activation.analysis
"""

from __future__ import annotations

from pathlib import Path

import decode_area_latency

from fusion import common
from fusion.up_gate_activation import model

TITLE = "Fusion 4: up_gate + SwiGLU activation"
RESULT_DIR = Path(__file__).resolve().parent / "result"


def run() -> common.FusionComparison:
    return common.run_fusion(
        model, decode_area_latency, TITLE, RESULT_DIR, "up_gate_activation_sweep.csv"
    )


if __name__ == "__main__":
    run()
