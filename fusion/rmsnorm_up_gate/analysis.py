"""Decode area-distribution analysis — Fusion 3: RMSNorm + up_gate.

Run: conda run -n fusion python -m fusion.rmsnorm_up_gate.analysis
"""

from __future__ import annotations

from pathlib import Path

import decode_area_latency

from fusion import common
from fusion.rmsnorm_up_gate import model

TITLE = "Fusion 3: pre-FFN RMSNorm + up_gate"
RESULT_DIR = Path(__file__).resolve().parent / "result"


def run() -> common.FusionComparison:
    return common.run_fusion(
        model, decode_area_latency, TITLE, RESULT_DIR, "rmsnorm_up_gate_sweep.csv"
    )


if __name__ == "__main__":
    run()
