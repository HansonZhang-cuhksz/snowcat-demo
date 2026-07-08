"""Latency-aware Snowcat-roofline execution-time estimator for a single GEMM.

Estimates the wall-clock time of one GEMM on a real GPU (RTX 4060 Laptop by default)
given the GEMM size and a fully specified mapping (tile size, software-pipelining
stages, loop order).  No GEMM is run on the GPU -- the estimate is analytical.

Two tools, reused from the area studies (see notes/single_gemm_estimator.md):

  1. Snowcat / Orojenesis traffic model  (snowcat_demo.model.traffic)
       W = buffer_bytes  -> one-stage SMEM working set of the tiling
       T = total_bytes   -> minimum HBM backing-store traffic of the tiling
  2. Latency-aware roofline
       latency  = HBM_LATENCY_CYCLES / CLOCK_HZ
       inflight = num_sm * C * W                       (Little's law, chip level)
       BW_eff   = min(BW_physical, inflight / latency)
       time     = max(ops / peak_tensor_flops,  T / BW_eff)

Usage:
  conda run -n profiling python gemm_time_estimator.py \
      --m 128 --n 4096 --k 6144 --bm 64 --bn 128 --bk 64 --order MKN --stages 2

  # auto-pick the smallest-optimal pipeline depth C:
  conda run -n profiling python gemm_time_estimator.py --m 128 --n 4096 --k 6144 \
      --bm 64 --bn 128 --bk 64 --order MKN

  # run the built-in decode-FFN example set:
  conda run -n profiling python gemm_time_estimator.py --demo
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.pareto import best_at_capacity
from snowcat_demo.model.traffic import LOOP_ORDERS, estimate_mapping_traffic
from snowcat_demo.model.workload import GemmWorkload, divisors


# --------------------------------------------------------------------------- #
# GPU model                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GpuModel:
    """Fixed hardware description used by the roofline.

    All fields are spec-sheet / device-queried constants -- edit them to model a
    different GPU.  Derived rooflines are computed as properties.
    """

    name: str
    num_sm: int
    tensor_cores: int
    tensor_flops_per_core_per_clock: float  # dense FP16/BF16 with FP32 accumulate
    clock_hz: float                         # SM clock used for both compute + latency
    bw_bytes_per_s: float                   # physical HBM/GDDR bandwidth (whole chip)
    smem_per_block_bytes: int               # usable shared memory for one threadblock
    smem_per_sm_bytes: int                  # total shared memory per SM (context)
    hbm_latency_cycles: float               # round-trip global-memory latency
    bytes_per_element: int = 2              # BF16 / FP16

    @property
    def peak_tensor_flops(self) -> float:
        """Chip-wide tensor-core compute roof (FLOP/s)."""
        return self.tensor_cores * self.tensor_flops_per_core_per_clock * self.clock_hz

    @property
    def latency_seconds(self) -> float:
        return self.hbm_latency_cycles / self.clock_hz


# RTX 4060 Laptop GPU (AD107, Ada, compute capability 8.9).
# SM count / SMEM sizes queried live from torch.cuda.get_device_properties(0);
# clock from `nvidia-smi --query-gpu=clocks.max.sm`; tensor rate + bandwidth from
# the Ada spec sheet.  See notes/single_gemm_estimator.md for provenance.
RTX4060_LAPTOP = GpuModel(
    name="NVIDIA GeForce RTX 4060 Laptop GPU",
    num_sm=24,
    tensor_cores=96,                          # 4 4th-gen tensor cores per SM
    tensor_flops_per_core_per_clock=512.0,    # dense FP16, FP32 accumulate (A100-derived)
    clock_hz=3105e6,                          # max SM boost; laptop sustains less (editable)
    bw_bytes_per_s=256e9,                      # GDDR6, 128-bit, 16 Gbps effective
    smem_per_block_bytes=101376,              # torch shared_memory_per_block_optin
    smem_per_sm_bytes=102400,                 # torch shared_memory_per_multiprocessor
    hbm_latency_cycles=500,                   # placeholder (same as A100 area study)
    bytes_per_element=2,
)

GPUS: dict[str, GpuModel] = {"rtx4060-laptop": RTX4060_LAPTOP}


# --------------------------------------------------------------------------- #
# Mapping                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Mapping:
    """A fully specified GEMM tiling.

    bm/bn/bk are the tile sizes (m0/n0/k0 in Snowcat terms) and must divide M/N/K.
    loop_order is the tile-loop nesting, outermost first, e.g. ("M", "K", "N").
    num_stages is the software-pipeline depth C; None -> auto-pick smallest optimal.
    """

    bm: int
    bn: int
    bk: int
    loop_order: tuple[str, str, str]
    num_stages: int | None = None


def optimal_mapping(m: int, n: int, k: int, gpu: GpuModel) -> Mapping:
    """The snowcat min-HBM-traffic tiling that fits one threadblock's SMEM budget.

    Convenience for comparing against a library kernel (cuBLAS also picks a good
    mapping).  This is a *search* over the snowcat mapspace, unlike the rest of the
    estimator which takes the mapping as a given input.
    """
    workload = GemmWorkload(m=m, k=k, n=n, bytes_per_element=gpu.bytes_per_element)
    best = best_at_capacity(enumerate_mappings(workload), gpu.smem_per_block_bytes)
    if best is None:
        raise ValueError("no mapping fits the SMEM budget")
    mp = best.mapping
    return Mapping(bm=mp.m0, bn=mp.n0, bk=mp.k0, loop_order=mp.loop_order)


def parse_loop_order(text: str) -> tuple[str, str, str]:
    """Accept 'MKN' or 'M-K-N' / 'M,K,N' -> ('M', 'K', 'N')."""
    cleaned = text.upper().replace("-", "").replace(",", "").replace(" ", "")
    order = tuple(cleaned)
    if len(order) != 3 or set(order) != {"M", "K", "N"}:
        raise ValueError(f"loop order must be a permutation of M,K,N; got {text!r}")
    if order not in LOOP_ORDERS:
        raise ValueError(f"unsupported loop order {order}; must be one of {LOOP_ORDERS}")
    return order  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Estimation                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Estimate:
    # inputs
    m: int
    n: int
    k: int
    mapping: Mapping
    gpu: GpuModel
    # snowcat traffic
    ops: int
    working_set_bytes: int          # W = buffer_bytes (one pipeline stage)
    traffic_bytes: int              # T = HBM backing-store traffic
    operational_intensity: float    # OI = ops / T  (FLOP/byte)
    # pipeline / latency
    num_stages: int                 # C actually used
    max_feasible_stages: int        # floor(SMEM_per_block / W)
    inflight_bytes: float           # num_sm * C * W
    bw_eff_bytes_per_s: float
    # roofline
    compute_time_s: float
    memory_time_s: float
    time_s: float
    bottleneck: str
    # wave-quantization diagnostics
    output_tiles: int
    waves: int
    sm_utilization: float
    wave_adjusted_time_s: float
    # feasibility
    fits_smem: bool
    notes: list[str] = field(default_factory=list)

    @property
    def effective_tflops(self) -> float:
        return self.ops / self.time_s / 1e12 if self.time_s > 0 else float("nan")


def _auto_num_stages(gpu: GpuModel, w: int) -> tuple[int, int]:
    """Smallest-optimal pipeline depth C and the max feasible depth (notes model)."""
    c_max = gpu.smem_per_block_bytes // w
    if c_max < 1:
        return 0, 0
    # BW saturates when num_sm * C * W / latency >= bw
    c_sat = math.ceil(gpu.bw_bytes_per_s * gpu.latency_seconds / (gpu.num_sm * w))
    c_best = min(c_max, max(c_sat, 1))
    return c_best, c_max


def estimate_gemm_time(
    m: int,
    n: int,
    k: int,
    mapping: Mapping,
    gpu: GpuModel = RTX4060_LAPTOP,
) -> Estimate:
    """Latency-aware Snowcat-roofline time estimate for one GEMM + mapping."""
    for name, dim, tile in (("M", m, mapping.bm), ("N", n, mapping.bn), ("K", k, mapping.bk)):
        if tile <= 0 or dim % tile != 0:
            raise ValueError(
                f"tile {name}0={tile} must be a positive divisor of {name}={dim}; "
                f"nearest divisors: {divisors(dim)}"
            )

    workload = GemmWorkload(m=m, k=k, n=n, bytes_per_element=gpu.bytes_per_element)
    traffic = estimate_mapping_traffic(
        workload, mapping.bm, mapping.bk, mapping.bn, mapping.loop_order
    )
    w = traffic.buffer_bytes          # one-stage working set W
    t = traffic.total_bytes           # HBM traffic T
    ops = workload.operations
    oi = ops / t

    notes: list[str] = []

    # ---- pipeline depth C -------------------------------------------------- #
    c_best_auto, c_max = _auto_num_stages(gpu, w)
    if mapping.num_stages is None:
        c = c_best_auto
        if c == 0:
            notes.append(
                f"working set W={w} B exceeds SMEM/block={gpu.smem_per_block_bytes} B; "
                "even C=1 does not fit."
            )
            c = 1
    else:
        c = mapping.num_stages
        if c < 1:
            raise ValueError("num_stages must be >= 1")

    fits_smem = c * w <= gpu.smem_per_block_bytes
    if not fits_smem:
        notes.append(
            f"C*W = {c * w} B exceeds SMEM/block = {gpu.smem_per_block_bytes} B "
            f"(max feasible C = {c_max}); pipeline would not fit on hardware."
        )

    # ---- latency-aware effective bandwidth (Little's law) ------------------ #
    inflight = gpu.num_sm * c * w
    bw_eff = min(gpu.bw_bytes_per_s, inflight / gpu.latency_seconds)

    # ---- roofline ---------------------------------------------------------- #
    compute_time = ops / gpu.peak_tensor_flops
    memory_time = t / bw_eff
    time_s = max(compute_time, memory_time)
    if compute_time > memory_time:
        bottleneck = "compute"
    elif memory_time > compute_time:
        bottleneck = "memory"
    else:
        bottleneck = "balanced"

    # ---- wave-quantization diagnostics ------------------------------------ #
    output_tiles = (m // mapping.bm) * (n // mapping.bn)
    waves = math.ceil(output_tiles / gpu.num_sm)
    sm_util = output_tiles / (waves * gpu.num_sm)
    # only the compute roof is quantized by whole waves; memory time is unaffected.
    wave_adjusted_compute = compute_time / sm_util if sm_util > 0 else compute_time
    wave_adjusted_time = max(wave_adjusted_compute, memory_time)

    return Estimate(
        m=m, n=n, k=k, mapping=mapping, gpu=gpu,
        ops=ops, working_set_bytes=w, traffic_bytes=t, operational_intensity=oi,
        num_stages=c, max_feasible_stages=c_max,
        inflight_bytes=inflight, bw_eff_bytes_per_s=bw_eff,
        compute_time_s=compute_time, memory_time_s=memory_time,
        time_s=time_s, bottleneck=bottleneck,
        output_tiles=output_tiles, waves=waves, sm_utilization=sm_util,
        wave_adjusted_time_s=wave_adjusted_time,
        fits_smem=fits_smem, notes=notes,
    )


# --------------------------------------------------------------------------- #
# Reporting                                                                     #
# --------------------------------------------------------------------------- #
def format_estimate(e: Estimate) -> str:
    mib = 2 ** 20
    lines = [
        f"=== GEMM {e.m}x{e.n}x{e.k}  on  {e.gpu.name} ===",
        f"  ops                 : {e.ops / 1e9:.3f} GFLOP  ({e.ops / 1e12:.4f} TFLOP)",
        f"  tile (BM,BN,BK)     : ({e.mapping.bm}, {e.mapping.bn}, {e.mapping.bk})  "
        f"loop_order={'-'.join(e.mapping.loop_order)}",
        "",
        "  -- snowcat traffic model --",
        f"  working set W       : {e.working_set_bytes / 1024:.2f} KiB  "
        f"({e.working_set_bytes} B)  per pipeline stage",
        f"  HBM traffic T       : {e.traffic_bytes / mib:.3f} MiB  ({e.traffic_bytes} B)",
        f"  op. intensity OI    : {e.operational_intensity:.3f} FLOP/byte",
        "",
        "  -- latency-aware pipeline --",
        f"  num_stages C        : {e.num_stages}"
        + (f"  (auto; max feasible = {e.max_feasible_stages})"
           if e.mapping.num_stages is None
           else f"  (max feasible = {e.max_feasible_stages})"),
        f"  SMEM for C stages   : {e.num_stages * e.working_set_bytes / 1024:.2f} KiB "
        f"/ {e.gpu.smem_per_block_bytes / 1024:.2f} KiB per block"
        + ("" if e.fits_smem else "   *** DOES NOT FIT ***"),
        f"  in-flight bytes     : {e.inflight_bytes / mib:.3f} MiB "
        f"(num_sm={e.gpu.num_sm} x C={e.num_stages} x W)",
        f"  HBM latency         : {e.gpu.hbm_latency_cycles:g} cycles "
        f"({e.gpu.latency_seconds * 1e9:.1f} ns)",
        f"  effective BW        : {e.bw_eff_bytes_per_s / 1e9:.1f} GB/s "
        f"(physical {e.gpu.bw_bytes_per_s / 1e9:.0f} GB/s)",
        "",
        "  -- roofline --",
        f"  compute roof        : {e.gpu.peak_tensor_flops / 1e12:.2f} TFLOP/s",
        f"  compute time        : {e.compute_time_s * 1e3:.4f} ms",
        f"  memory time         : {e.memory_time_s * 1e3:.4f} ms",
        f"  >> time             : {e.time_s * 1e3:.4f} ms   "
        f"[{e.bottleneck}-bound]   {e.effective_tflops:.1f} TFLOP/s",
        "",
        "  -- wave quantization (diagnostic) --",
        f"  output tiles        : {e.output_tiles}  "
        f"(M/BM={e.m // e.mapping.bm} x N/BN={e.n // e.mapping.bn})",
        f"  waves               : {e.waves}  (num_sm={e.gpu.num_sm})   "
        f"SM utilization = {e.sm_utilization * 100:.1f}%",
        f"  wave-adjusted time  : {e.wave_adjusted_time_s * 1e3:.4f} ms",
    ]
    for note in e.notes:
        lines.append(f"  NOTE: {note}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Built-in decode-FFN example set (GLM-5.2, 4096-batched, 128 tok/expert)       #
# --------------------------------------------------------------------------- #
_DEMO_GEMMS = [
    # (label, M, N, K, BM, BN, BK, loop_order, stages)
    # Illustrative (not traffic-optimized) mappings that fit the 99 KiB SMEM budget.
    # For the small-M decode GEMMs, BM = full M keeps the weight streamed once.
    ("router",  4096, 256, 6144, 128, 128, 64, "MKN", None),
    ("up_gate",  128, 4096, 6144, 128, 128, 64, "MKN", None),
    ("down",     128, 6144, 2048, 128, 128, 64, "MKN", None),
]


def run_demo(gpu: GpuModel) -> None:
    for label, m, n, k, bm, bn, bk, order, stages in _DEMO_GEMMS:
        mapping = Mapping(bm=bm, bn=bn, bk=bk,
                          loop_order=parse_loop_order(order), num_stages=stages)
        e = estimate_gemm_time(m, n, k, mapping, gpu)
        print(f"\n### {label}")
        print(format_estimate(e))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", choices=sorted(GPUS), default="rtx4060-laptop")
    p.add_argument("--m", type=int, help="GEMM M")
    p.add_argument("--n", type=int, help="GEMM N")
    p.add_argument("--k", type=int, help="GEMM K")
    p.add_argument("--bm", type=int, help="tile BM (m0), must divide M")
    p.add_argument("--bn", type=int, help="tile BN (n0), must divide N")
    p.add_argument("--bk", type=int, help="tile BK (k0), must divide K")
    p.add_argument("--order", default="MKN", help="loop order, e.g. MKN or M-K-N")
    p.add_argument("--stages", type=int, default=None,
                   help="software-pipeline depth C (default: auto smallest-optimal)")
    p.add_argument("--optimal", action="store_true",
                   help="ignore --bm/--bn/--bk/--order and use the snowcat min-traffic "
                        "mapping that fits SMEM")
    p.add_argument("--clock-mhz", type=float, default=None,
                   help="override SM clock in MHz (e.g. a sustained laptop boost)")
    p.add_argument("--demo", action="store_true",
                   help="run the built-in decode-FFN example set")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gpu = GPUS[args.gpu]
    if args.clock_mhz is not None:
        from dataclasses import replace
        gpu = replace(gpu, clock_hz=args.clock_mhz * 1e6)

    if args.demo:
        run_demo(gpu)
        return

    required = ("m", "n", "k") if args.optimal else ("m", "n", "k", "bm", "bn", "bk")
    missing = [f for f in required if getattr(args, f) is None]
    if missing:
        raise SystemExit(
            f"missing required args: {', '.join('--' + x for x in missing)} "
            "(or pass --demo)"
        )

    if args.optimal:
        mapping = optimal_mapping(args.m, args.n, args.k, gpu)
        if args.stages is not None:
            mapping = Mapping(mapping.bm, mapping.bn, mapping.bk,
                              mapping.loop_order, args.stages)
    else:
        mapping = Mapping(
            bm=args.bm, bn=args.bn, bk=args.bk,
            loop_order=parse_loop_order(args.order), num_stages=args.stages,
        )
    e = estimate_gemm_time(args.m, args.n, args.k, mapping, gpu)
    print(format_estimate(e))


if __name__ == "__main__":
    main()
