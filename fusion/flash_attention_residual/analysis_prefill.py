"""Prefill area-distribution analysis — Fusion 1: FlashAttention (MLA output) + residual.

Full GLM-5.2 prefill layer (prefill_area_latency.py baseline, DSA attention), only the
mla_o_residual kernel fused. Run from the repo root:
    conda run -n fusion python -m fusion.flash_attention_residual.analysis_prefill
"""

from __future__ import annotations

from pathlib import Path

import prefill_area_latency

from fusion import common
from fusion.flash_attention_residual import model

TITLE = "Prefill Fusion 1: FlashAttention (MLA output) + residual add"
RESULT_DIR = Path(__file__).resolve().parent / "result"


def run() -> common.FusionComparison:
    return common.run_fusion(
        model, prefill_area_latency, TITLE, RESULT_DIR,
        "prefill_flash_attention_residual_sweep.csv", prefix="prefill_",
    )


if __name__ == "__main__":
    run()
