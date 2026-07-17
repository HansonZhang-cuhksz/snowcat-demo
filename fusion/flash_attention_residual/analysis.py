"""Decode area-distribution analysis — Fusion 1: FlashAttention (MLA output) + residual.

Run from the repo root:
    conda run -n fusion python -m fusion.flash_attention_residual.analysis
"""

from __future__ import annotations

from pathlib import Path

import decode_area_latency

from fusion import common
from fusion.flash_attention_residual import model

TITLE = "Fusion 1: FlashAttention (MLA output) + residual add"
RESULT_DIR = Path(__file__).resolve().parent / "result"


def run() -> common.FusionComparison:
    return common.run_fusion(
        model, decode_area_latency, TITLE, RESULT_DIR,
        "flash_attention_residual_sweep.csv",
    )


if __name__ == "__main__":
    run()
