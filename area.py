from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ski_slope import min_attainable_traffic


# Chip constants. Keep these aligned with ffn_area_latency.py for apples-to-apples runs.
# A_total = 694.116 * 10**6              # um^2
A_total = 136.29 * 10**6              # um^2
A_bit = 0.0864                         # um^2/bit
logic_density = 39.98                  # MTr/mm^2 == transistors/um^2
bw = 2.04 * 10**12                     # byte/s

num_sm = 1
A_total = A_total / num_sm
bw = bw / num_sm

# Architecture placeholders. Replace these with measured/spec-sheet values.
CUDA_CORE_TRANSISTORS = 0.2 * 10**6     # transistors/CUDA core placeholder
TENSOR_CORE_TRANSISTORS = 6.0 * 10**6   # transistors/tensor core placeholder
TENSOR_FLOPS = 512 * 1.00 * 10**9       # flops/s/tensor core placeholder
CUDA_CLOCK_HZ = 1410 * 10**6            # Hz
ACTIVATION_FLOPS_PER_CUDA_CORE = 5.64 * 10 ** 9      # flops/s/CUDA core placeholder

A_cuda_core = CUDA_CORE_TRANSISTORS / logic_density
A_tensor_core = TENSOR_CORE_TRANSISTORS / logic_density

# GEMM task
M = 2048
N = 4096
K = 6144
BYTE_PER_ELEMENT = 2


def min_traffic_at_capacity(capacity):
    if capacity < 3 * BYTE_PER_ELEMENT:
        return np.nan
    return min_attainable_traffic(
        (M, N, K),
        bytes_per_element=BYTE_PER_ELEMENT,
        sram_capacity_bytes=int(capacity),
    )


def _save_plot(r, curve, label, title, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(r, curve, label=label)
    plt.xlabel("SRAM Utilization")
    plt.ylabel("Performance (FLOPS)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    Path("./result").mkdir(parents=True, exist_ok=True)

    r = np.arange(0, 1, 0.001)

    sram_bytes = r * A_total / A_bit / 8    # bytes
    tensors = np.floor((1-r) * A_total / A_tensor_core)    # tensors

    operations = 2 * M * N * K              # flops
    with ProcessPoolExecutor() as executor:
        min_traffic = np.fromiter(
            executor.map(min_traffic_at_capacity, sram_bytes, chunksize=25),
            dtype=float,
            count=len(sram_bytes),
        )                                   # bytes
    oi = operations / min_traffic           # flops/byte
    p_mem = oi * bw                         # flops/s
    p_compute = tensors * TENSOR_FLOPS      # flops/s
    p_peak = np.minimum(p_mem, p_compute)   # flops/s

    best_index = int(np.nanargmax(p_peak))
    if p_mem[best_index] < p_compute[best_index]:
        bottleneck = "memory"
    elif p_compute[best_index] < p_mem[best_index]:
        bottleneck = "compute"
    else:
        bottleneck = "balanced"

    print("=== Configuration ===")
    print(f"Chip area: {A_total:.6g} um^2 (num_sm={num_sm})")
    print(f"SMEM bit area: {A_bit} um^2/bit")
    print(f"Logic density: {logic_density} MTr/mm^2")
    print(f"HBM bandwidth: {bw / 1e12:.6g} TB/s")
    print(f"Tensor core transistors: {TENSOR_CORE_TRANSISTORS / 1e6:g} MTr")
    print(f"Tensor core FLOPS: {TENSOR_FLOPS / 1e9:g} GFLOP/s")
    print(f"GEMM: M={M}, N={N}, K={K}")
    print(f"Operations: {operations / 1e12:.6g} TFLOP")
    print(f"Area grid step: 0.001")

    print("\n=== Best Area Point ===")
    print(f"Best SMEM fraction (r): {r[best_index]:.3f}")
    print(f"SMEM: {sram_bytes[best_index] / 2**20:.3f} MiB")
    print(f"Tensor cores: {int(tensors[best_index])}")
    print(f"Peak performance: {p_peak[best_index] / 1e12:.6f} TFLOP/s")
    print(f"Memory performance: {p_mem[best_index] / 1e12:.6f} TFLOP/s")
    print(f"Compute performance: {p_compute[best_index] / 1e12:.6f} TFLOP/s")
    print(f"Bottleneck: {bottleneck}")

    _save_plot(
        r, p_peak, "Peak Performance",
        "Performance vs. SRAM Utilization", "./result/p-r_peak.png",
    )
    _save_plot(
        r, p_mem, "Memory Performance",
        "Performance vs. SRAM Utilization", "./result/p-r_mem.png",
    )
    _save_plot(
        r, p_compute, "Compute Performance",
        "Performance vs. SRAM Utilization", "./result/p-r_compute.png",
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(r, p_peak, label="Peak Performance")
    plt.plot(r, p_mem, label="Memory Performance")
    plt.plot(r, p_compute, label="Compute Performance")
    plt.xlabel("SRAM Utilization")
    plt.ylabel("Performance (FLOPS)")
    plt.title("Performance vs. SRAM Utilization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./result/p-r.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
