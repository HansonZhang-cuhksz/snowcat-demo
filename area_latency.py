from __future__ import annotations

import numpy as np

from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.workload import GemmWorkload


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

# Pipeline-depth / latency-hiding model.  num_stages (C) is solved per kernel:
# each concurrent task occupies one Snowcat tile working set (W = buffer_bytes) in
# SMEM, and C tasks stay in flight to hide HBM latency (BW_eff = min(bw, C*W/latency)).
HBM_LATENCY_CYCLES = 500
HBM_CLOCK_HZ = 1215 * 10**6
LATENCY_SECONDS = HBM_LATENCY_CYCLES / CUDA_CLOCK_HZ

# GEMM task
M = 2048
N = 4096
K = 6144
BYTE_PER_ELEMENT = 2


def build_frontier() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[str, str, str], ...]]:
    """Pareto frontier of (one-stage working set W, minimum HBM traffic T) for the
    GEMM, keyed on W = buffer_bytes, together with the achieving tile/loop_order.
    Mirrors ffn_area_latency.py's frontier build (no tensor-core tile filter,
    matching area.py's use of all mappings)."""
    workload = GemmWorkload(
        m=M, k=K, n=N, bytes_per_element=BYTE_PER_ELEMENT
    )
    pairs = sorted(
        (
            point.buffer_bytes,
            point.backing_store_bytes,
            point.mapping.m0,
            point.mapping.n0,
            point.mapping.k0,
            point.mapping.loop_order,
        )
        for point in enumerate_mappings(workload)
    )
    frontier_w: list[int] = []
    frontier_t: list[int] = []
    frontier_bm: list[int] = []
    frontier_bn: list[int] = []
    frontier_bk: list[int] = []
    frontier_lo: list[tuple[str, str, str]] = []
    best: tuple[int, int, int, int, tuple[str, str, str]] | None = None

    for w, t, bm, bn, bk, lo in pairs:
        if best is None or t < best[0]:
            best = (t, bm, bn, bk, lo)
        if frontier_w and w == frontier_w[-1]:
            frontier_t[-1] = best[0]
            frontier_bm[-1] = best[1]
            frontier_bn[-1] = best[2]
            frontier_bk[-1] = best[3]
            frontier_lo[-1] = best[4]
        else:
            frontier_w.append(w)
            frontier_t.append(best[0])
            frontier_bm.append(best[1])
            frontier_bn.append(best[2])
            frontier_bk.append(best[3])
            frontier_lo.append(best[4])

    w_arr = np.array(frontier_w, dtype=np.int64)
    t_arr = np.array(frontier_t, dtype=np.int64)
    improved = np.r_[True, t_arr[1:] < t_arr[:-1]]
    return (
        w_arr[improved],
        t_arr[improved],
        np.array(frontier_bm, dtype=np.int64)[improved],
        np.array(frontier_bn, dtype=np.int64)[improved],
        np.array(frontier_bk, dtype=np.int64)[improved],
        tuple(lo for keep, lo in zip(improved, frontier_lo) if keep),
    )


def select_mapping(
    frontier_w: np.ndarray,
    frontier_t: np.ndarray,
    frontier_bm: np.ndarray,
    frontier_bn: np.ndarray,
    frontier_bk: np.ndarray,
    frontier_lo: tuple[tuple[str, str, str], ...],
    s_total: float,
    tensor_roof: float,
    operations: int,
) -> dict[str, object] | None:
    """Winning tiling + num_stages at a single (fixed) SMEM budget."""
    compute_time = operations / tensor_roof if tensor_roof > 0 else np.inf
    best: dict[str, object] | None = None
    for i in range(len(frontier_w)):
        w_i = int(frontier_w[i])
        t_i = int(frontier_t[i])
        c_max = int(s_total // w_i)
        if c_max < 1:
            continue
        c_sat = int(np.ceil(bw * LATENCY_SECONDS / w_i))
        c_best = min(c_max, c_sat)
        bw_eff = min(bw, c_best * w_i / LATENCY_SECONDS)
        mem_time = t_i / bw_eff
        time_i = max(compute_time, mem_time)
        if best is None or time_i < best["time"]:  # type: ignore[typeddict-item]
            best = {
                "bm": int(frontier_bm[i]),
                "bn": int(frontier_bn[i]),
                "bk": int(frontier_bk[i]),
                "loop_order": frontier_lo[i],
                "num_stages": c_best,
                "max_feasible_stages": c_max,
                "one_stage_smem": w_i,
                "traffic": t_i,
                "oi": operations / t_i,
                "bw_eff": bw_eff,
                "time": time_i,
            }
    return best


def performance(
    frontier_w: np.ndarray,
    frontier_t: np.ndarray,
    s_total: np.ndarray,
    tensor_roof: np.ndarray,
    operations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latency-aware performance per area node.

    For each frontier point i (W_i, T_i) the optimal pipeline depth is
    C_best = min(floor(S_total / W_i), ceil(bw * latency / W_i)) and
    BW_eff = min(bw, C_best * W_i / latency); memory time = T_i / BW_eff.

    Because compute time is common across points, the peak is
    p_peak = min(p_mem, p_compute) where
    p_mem = operations / min_i(T_i / BW_eff_i) and p_compute = tensor_roof.
    """
    n = len(s_total)
    p_compute = tensor_roof
    mem_time_best = np.full(n, np.inf, dtype=float)
    for i in range(len(frontier_w)):
        w_i = float(frontier_w[i])
        t_i = float(frontier_t[i])
        c_max = np.floor(s_total / w_i)                 # max C with S_eff >= W_i
        valid = c_max >= 1
        c_sat = int(np.ceil(bw * LATENCY_SECONDS / w_i))  # smallest C saturating bw
        c_best = np.minimum(c_max, c_sat)
        c_safe = np.where(valid, c_best, 1.0)
        bw_eff = np.minimum(bw, c_safe * w_i / LATENCY_SECONDS)
        with np.errstate(divide="ignore", invalid="ignore"):
            mem_time = t_i / bw_eff
        mem_time = np.where(valid, mem_time, np.inf)
        mem_time_best = np.minimum(mem_time_best, mem_time)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_mem = operations / mem_time_best
    p_peak = np.minimum(p_mem, p_compute)
    return p_peak, p_mem, p_compute


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


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    Path("./result").mkdir(parents=True, exist_ok=True)

    r = np.arange(0, 1, 0.001)

    sram_bytes = r * A_total / A_bit / 8               # total SMEM budget S_total
    tensors = np.floor((1 - r) * A_total / A_tensor_core)  # tensor-core count
    operations = 2 * M * N * K                         # flops
    tensor_roof = tensors * TENSOR_FLOPS               # flops/s

    frontier = build_frontier()
    frontier_w, frontier_t, frontier_bm, frontier_bn, frontier_bk, frontier_lo = frontier
    p_peak, p_mem, p_compute = performance(
        frontier_w, frontier_t, sram_bytes, tensor_roof, operations
    )

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
    print(f"CUDA clock: {CUDA_CLOCK_HZ / 1e6:g} MHz")
    print(f"HBM latency: {HBM_LATENCY_CYCLES} cycles ({LATENCY_SECONDS * 1e9:.3f} ns)")
    print(f"GEMM: M={M}, N={N}, K={K}")
    print(f"Operations: {operations / 1e12:.6g} TFLOP")
    print(f"Pareto frontier points: {len(frontier_w)}")
    print(f"Area grid step: 0.001")

    print("\n=== Best Area Point ===")
    print(f"Best SMEM fraction (r): {r[best_index]:.3f}")
    print(f"SMEM: {sram_bytes[best_index] / 2**20:.3f} MiB")
    print(f"Tensor cores: {int(tensors[best_index])}")
    print(f"Peak performance: {p_peak[best_index] / 1e12:.6f} TFLOP/s")
    print(f"Memory performance: {p_mem[best_index] / 1e12:.6f} TFLOP/s")
    print(f"Compute performance: {p_compute[best_index] / 1e12:.6f} TFLOP/s")
    print(f"Bottleneck: {bottleneck}")

    print("\n=== Optimal Mapping (at best area point) ===")
    mapping = select_mapping(
        frontier_w, frontier_t, frontier_bm, frontier_bn, frontier_bk, frontier_lo,
        float(sram_bytes[best_index]), float(tensor_roof[best_index]), operations,
    )
    if mapping is None:
        print("  no mapping fits selected SMEM capacity")
    else:
        print(
            f"  BM={mapping['bm']}, BN={mapping['bn']}, BK={mapping['bk']}, "
            f"loop_order={'-'.join(mapping['loop_order'])}, "
            f"num_stages={mapping['num_stages']} "
            f"(max_feasible={mapping['max_feasible_stages']}), "
            f"one_stage_smem={mapping['one_stage_smem'] / 2**20:.6f} MiB "
            f"({mapping['one_stage_smem']} bytes), "
            f"traffic={mapping['traffic'] / 2**20:.3f} MiB, "
            f"OI={mapping['oi']:.6f} FLOP/byte, "
            f"BW_eff={mapping['bw_eff'] / 1e12:.6f} TB/s"
        )

    _save_plot(
        r, p_peak, "Peak Performance",
        "Performance vs. SRAM Utilization (latency-aware)",
        "./result/p-r_latency_peak.png",
    )
    _save_plot(
        r, p_mem, "Memory Performance",
        "Performance vs. SRAM Utilization (latency-aware)",
        "./result/p-r_latency_mem.png",
    )
    _save_plot(
        r, p_compute, "Compute Performance",
        "Performance vs. SRAM Utilization",
        "./result/p-r_latency_compute.png",
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(r, p_peak, label="Peak Performance")
    plt.plot(r, p_mem, label="Memory Performance")
    plt.plot(r, p_compute, label="Compute Performance")
    plt.xlabel("SRAM Utilization")
    plt.ylabel("Performance (FLOPS)")
    plt.title("Performance vs. SRAM Utilization (latency-aware)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./result/p-r_latency.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
