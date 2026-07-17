"""Prefill area-distribution analysis — Fusion 3: RMSNorm + up_gate.

Run: conda run -n fusion python -m fusion.rmsnorm_up_gate.analysis_prefill
"""

from __future__ import annotations

from pathlib import Path

import prefill_area_latency

from fusion import common
from fusion.rmsnorm_up_gate import model

TITLE = "Prefill Fusion 3: pre-FFN RMSNorm + up_gate"
RESULT_DIR = Path(__file__).resolve().parent / "result"


def run() -> common.FusionComparison:
    return common.run_fusion(
        model, prefill_area_latency, TITLE, RESULT_DIR,
        "prefill_rmsnorm_up_gate_sweep.csv", prefix="prefill_",
    )


if __name__ == "__main__":
    run()
