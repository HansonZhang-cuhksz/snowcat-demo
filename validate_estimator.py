"""Optional accuracy check: estimator vs. measured torch.matmul (needs a GPU + torch).

Compares gemm_time_estimator's latency-aware snowcat-roofline prediction against a real
timed FP16 matmul.  The estimate is only comparable to a cuBLAS kernel when fed a
near-optimal mapping, so we use the snowcat min-traffic mapping (`--optimal`) here.

  conda run -n profiling python validate_estimator.py
"""
from __future__ import annotations

import torch

from gemm_time_estimator import RTX4060_LAPTOP, estimate_gemm_time, optimal_mapping


def measure(m: int, n: int, k: int, iters: int = 200, warmup: int = 50) -> float:
    """Mean wall-clock ms of one FP16 (m,k)@(k,n) matmul via CUDA events."""
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


SHAPES = [
    ("router",   4096, 256, 6144),
    ("up_gate",   128, 4096, 6144),
    ("down",      128, 6144, 2048),
    ("square2k", 2048, 2048, 2048),
    ("big",      4096, 4096, 4096),
]


def main() -> None:
    gpu = RTX4060_LAPTOP
    print(f"GPU: {gpu.name}   compute roof {gpu.peak_tensor_flops/1e12:.1f} TFLOP/s   "
          f"BW {gpu.bw_bytes_per_s/1e9:.0f} GB/s")
    print(f"{'gemm':<9} {'MxNxK':<18} {'tile(BM,BN,BK)':<16} {'est ms':>8} "
          f"{'meas ms':>8} {'est/meas':>9} {'bottleneck':>11}")
    for label, m, n, k in SHAPES:
        mp = optimal_mapping(m, n, k, gpu)
        e = estimate_gemm_time(m, n, k, mp, gpu)
        meas = measure(m, n, k)
        tile = f"({mp.bm},{mp.bn},{mp.bk})"
        print(f"{label:<9} {f'{m}x{n}x{k}':<18} {tile:<16} {e.time_s*1e3:>8.4f} "
              f"{meas:>8.4f} {e.time_s*1e3/meas:>9.2f} {e.bottleneck:>11}")


if __name__ == "__main__":
    main()
