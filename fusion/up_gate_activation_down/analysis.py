"""Decode area-distribution analysis — Fusion 6: up_gate + SwiGLU + down (full FFN).

Run: conda run -n fusion python -m fusion.up_gate_activation_down.analysis
"""

from __future__ import annotations

from pathlib import Path

import decode_area_latency

from fusion import common
from fusion.up_gate_activation_down import model

TITLE = "Fusion 6: up_gate + SwiGLU + down (full FFN fusion)"
RESULT_DIR = Path(__file__).resolve().parent / "result"


def run() -> common.FusionComparison:
    return common.run_fusion(
        model, decode_area_latency, TITLE, RESULT_DIR,
        "up_gate_activation_down_sweep.csv",
    )


if __name__ == "__main__":
    run()
