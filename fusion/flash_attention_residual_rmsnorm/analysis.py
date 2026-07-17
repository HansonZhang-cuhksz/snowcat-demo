"""Decode area-distribution analysis — Fusion 2: FA + residual + RMSNorm.

Run: conda run -n fusion python -m fusion.flash_attention_residual_rmsnorm.analysis
"""

from __future__ import annotations

from pathlib import Path

import decode_area_latency

from fusion import common
from fusion.flash_attention_residual_rmsnorm import model

TITLE = "Fusion 2: FlashAttention (MLA output) + residual add + RMSNorm"
RESULT_DIR = Path(__file__).resolve().parent / "result"


def run() -> common.FusionComparison:
    return common.run_fusion(
        model, decode_area_latency, TITLE, RESULT_DIR,
        "flash_attention_residual_rmsnorm_sweep.csv",
    )


if __name__ == "__main__":
    run()
