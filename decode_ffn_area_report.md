# Decode FFN Area-Balance Report

This report summarizes the area-distribution model built in this repository for
the decode-stage MoE FFN pass.  The goal is to balance area among CUDA cores,
tensor cores, and SMEM, then quantify how CODA-style GEMM-epilogue fusion
changes the optimum.

The final comparison uses the conservative fused model in `ffn_fused_area.py`:
router and up-gate use CODA-style fusion, while down GEMM, top-k weighted sum,
and residual add remain separate.  The optimistic top-k
`down_weighted_sum_residual` fusion is not used because a standard CODA epilogue
only sees one GEMM output tile and cannot generally hold all top-k expert outputs
for one token on chip.

## Assumptions
1. Workload is 8192-batched GLM-5.2
    - Experts: 512
    - Router Top-K: 8
    - Hidden Size: 6144
    - Intermediate Size: 2048
2. BF16 Weight & Activation
3. Tokens are evenly distributed across experts
    - 128 tokens per expert
4. TSMC-12FFC logic node
    - 39.98 MTr/mm²
5. TSMC-N12-SHC SRAM node
    - 0.0864 μm²/bit
6. Transistor per CUDA core ≈ A100
    - 0.2 MTr per CUDA core
7. Transistor per Tensor core ≈ A100
    - 6.0 MTr per Tensor core
8. Compute Power per CUDA core ≈ A100
    - 5.64 GFLOP/s
9. Compute Power per Tensor core ≈ A100
    - 512 GFLOP/s
10. Clock Frequency ≈ A100
    - 1410 MHz
11. SwiGLU FLOPS per element: 8

## Workload

The modeled decode FFN pass uses:

| Parameter | Value |
|---|---:|
| Batch tokens | 8192 |
| Experts | 512 |
| Tokens per expert | 128 |
| Hidden size | 6144 |
| Intermediate size | 2048 |
| Router top-k | 8 |
| Element size | 2 bytes |

The top-k follows from:

```text
top_k = experts * tokens_per_expert / batch_tokens
      = 512 * 128 / 8192
      = 8
```

## Hardware-Area Model

Current constants:

| Quantity | Formula | Value |
|---|---:|---:|
| Total modeled area | `A_total` | 136.29 mm^2 |
| SRAM bit area | `A_bit` | 0.0864 um^2/bit |
| Logic density | `logic_density` | 39.98 MTr/mm^2 |
| HBM bandwidth | `bw` | 2.04 TB/s |
| CUDA core transistors | `CUDA_CORE_TRANSISTORS` | 0.2 MTr/core |
| Tensor core transistors | `TENSOR_CORE_TRANSISTORS` | 6.0 MTr/core |
| Tensor throughput | `TENSOR_FLOPS` | 512 GFLOP/s/core |
| CUDA throughput | `ACTIVATION_FLOPS_PER_CUDA_CORE` | 5.64 GFLOP/s/core |
| CUDA clock | `CUDA_CLOCK_HZ` | 1410 MHz |

Area per core:

```text
A_cuda_core = CUDA_CORE_TRANSISTORS / logic_density
            = 0.2e6 / 39.98
            = 5002.501 um^2

A_tensor_core = TENSOR_CORE_TRANSISTORS / logic_density
              = 6.0e6 / 39.98
              = 150075.038 um^2
```

Area split:

```text
rc = CUDA-core area fraction
rt = tensor-core area fraction
r_smem = 1 - rc - rt
```

Converted resources:

```text
cuda_cores  = floor(rc * A_total / A_cuda_core)
tensor_cores = floor(rt * A_total / A_tensor_core)
smem_bytes = r_smem * A_total / A_bit / 8

cuda_roof = cuda_cores * ACTIVATION_FLOPS_PER_CUDA_CORE
tensor_roof = tensor_cores * TENSOR_FLOPS
```

## Timing Model

For GEMMs, the Snowcat/Orojenesis-style traffic model enumerates divisor tile
sizes and loop orders.  For each mapping, the model computes backing-store/HBM
traffic under a one-tile cache model.  At each SMEM capacity, the minimum
attainable traffic is selected.

For an unfused GEMM:

```text
GEMM_ops = 2 * M * N * K
OI = GEMM_ops / min_HBM_traffic
memory_roof = OI * bw
attainable = min(memory_roof, tensor_roof)
time = GEMM_ops / attainable
```

For vector/reduction kernels:

```text
OI = vector_ops / vector_HBM_traffic
memory_roof = OI * bw
attainable = min(memory_roof, cuda_roof)
time = vector_ops / attainable
```

For fused CODA GEMM-epilogue kernels, the model uses a three-way roof:

```text
time_fused = max(
    tensor_GEMM_ops / tensor_roof,
    epilogue_CUDA_ops / cuda_roof,
    fused_HBM_traffic / bw
)
```

This is stricter than collapsing everything into a single OI because tensor-core
mainloop work and CUDA/SFU epilogue work use different hardware.

## Unfused Workload

The unfused pass is:

```text
RMSNorm square-reduction
router GEMM
up_gate GEMM x512
SwiGLU activation
down GEMM x512
top-k expert weighted sum
residual add
```

### GEMMs

| Stage | Shape | Count | Ops |
|---|---:|---:|---:|
| router | M=8192, N=512, K=6144 | 1 | 51.540 GFLOP |
| up_gate | M=128, N=4096, K=6144 | 512 | 3298.535 GFLOP |
| down | M=128, N=6144, K=2048 | 512 | 1649.267 GFLOP |

### Vector/Reduction Stages

| Stage | Formula | Ops | HBM traffic | OI |
|---|---|---:|---:|---:|
| RMSNorm square-reduction | square + hidden reduction over 8192x6144 | 100.655 MFLOP | 96.031 MiB | 0.9996 |
| activation | `512*128*2048*8` | 1073.742 MFLOP | 768.000 MiB | 1.3333 |
| expert weighted sum | top-k multiply-add over all batch tokens | 754.975 MFLOP | 864.125 MiB | 0.8332 |
| residual add | one add over 8192x6144 | 50.332 MFLOP | 288.000 MiB | 0.1667 |

The top-k weighted sum is modeled over all batched tokens:

```text
ops = batch * top_k * hidden          # multiplies
    + batch * (top_k - 1) * hidden    # adds
    = 754,974,720 FLOP

traffic = batch * top_k * hidden * 2      # expert-output reads
        + batch * top_k * 2               # gate reads
        + batch * hidden * 2              # output writes
        = 906,100,736 bytes
        = 864.125 MiB
```

## Conservative Fused Workload

The conservative fused pass is:

```text
RMSNorm square-reduction
router + RMS-scale epilogue
up_gate + RMS-scale + SwiGLU epilogue, x512
down GEMM x512
top-k expert weighted sum
residual add
```

The following fusions are used:

1. `router_rms_scale`
2. `up_gate_rms_swiglu_x512`

The following fusion is intentionally not used:

```text
down + weighted sum + residual
```

Reason: top-k combine is cross-expert.  A standard CODA epilogue sees the
current GEMM output tile, not all top-k expert output tiles for the same token.
A fully fused top-k down/combine/residual kernel would require a specialized
grouped or persistent schedule.  Therefore, the conservative model keeps down,
weighted sum, and residual as separate stages.

## Area-Balance Results

### Optimal Area Distribution

| Design | rc | rt | SMEM frac | SMEM MiB | CUDA cores | Tensor cores | Total time | Effective throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfused | 0.018 | 0.976 | 0.006 | 1.128 | 490 | 886 | 21.283 ms | 234.996 TFLOP/s |
| Conservative fused | 0.014 | 0.979 | 0.007 | 1.316 | 381 | 889 | 20.756 ms | 240.972 TFLOP/s |

Overall speedup:

```text
speedup = 21.282595 / 20.755932
        = 1.0254x
```

The optimum is tensor-core dominated in both cases because the runtime is mostly
from the up-gate and down GEMMs.  CUDA area drops in the fused case because the
standalone activation stage is folded into the up-gate epilogue.  SMEM rises
slightly from 0.006 to 0.007 because the fused up-gate epilogue has a slightly
larger tile footprint, and the best grid point shifts accordingly.

## Stage-Level Comparison

| Stage | Unfused counterpart | Unfused time | Fused time | Speedup | Unfused HBM | Fused HBM | Traffic ratio | Unfused OI | Fused OI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RMSNorm square-reduction | same | 0.049 ms | 0.049 ms | 1.00x | 96.031 MiB | 96.031 MiB | 1.00x | 1.000 | 1.000 |
| router_rms_scale | router | 0.114 ms | 0.113 ms | 1.00x | 152.000 MiB | 152.016 MiB | 1.00x | 323.368 | 323.361 |
| up_gate_rms_swiglu_x512 | up_gate_x512 + activation | 13.685 ms | 13.159 ms | 1.04x | 26624.000 MiB | 25600.125 MiB | 1.04x | 118.192 | 122.929 |
| down_x512 | same | 6.842 ms | 6.842 ms | 1.00x | 13312.000 MiB | 13312.000 MiB | 1.00x | 118.154 | 118.154 |
| expert_weighted_sum | same | 0.444 ms | 0.444 ms | 1.00x | 864.125 MiB | 864.125 MiB | 1.00x | 0.833 | 0.833 |
| residual_add | same | 0.148 ms | 0.148 ms | 1.00x | 288.000 MiB | 288.000 MiB | 1.00x | 0.167 | 0.167 |

The only material gain in the conservative fused case is:

```text
up_gate_x512 + activation -> up_gate_rms_swiglu_x512
```

Traffic saved:

```text
26624.000 MiB - 25600.125 MiB = 1023.875 MiB
```

This corresponds to removing almost exactly one raw gate/up tensor round trip.
The runtime gain is modest because the stage remains mostly tensor-GEMM-bound.

## Min Attainable Traffic: Standard Snowcat Model

For a standard GEMM with problem `(M, N, K)` and tile `(m0, n0, k0)`:

```text
mt = M / m0
nt = N / n0
kt = K / k0

A_tile = m0 * k0 * bytes
W_tile = k0 * n0 * bytes
O_tile = m0 * n0 * bytes

buffer = A_tile + W_tile + O_tile
```

For each loop order in:

```text
(M,K,N), (M,N,K), (K,M,N), (K,N,M), (N,M,K), (N,K,M)
```

the model counts how many times each unique A tile and W tile is reloaded.  For
partial output accumulation, if K is inside the output-tile run, the output tile
stays on chip until complete:

```text
partial_reads = 0
partial_writes = 0
final_writes = mt * nt * O_tile
```

Otherwise partial accumulators spill:

```text
partial_reads = mt * nt * (kt - 1) * O_tile
partial_writes = mt * nt * (kt - 1) * O_tile
final_writes = mt * nt * O_tile
```

Minimum attainable traffic at capacity `C`:

```text
min_traffic(C) = min traffic(mapping)
                 over all divisor tiles and loop orders
                 where buffer(mapping) <= C
```

This is the principle used by `ffn_area.py`.

## Min Attainable Traffic: CODA Adaptations

The CODA traffic functions in `coda_fused_traffic.py` preserve the same
Snowcat search:

```text
enumerate divisor tiles
enumerate loop orders
reject mappings whose buffer > SMEM
take minimum HBM traffic
```

The adaptation changes only the epilogue-side output and auxiliary traffic.

### 1. GEMM + RMS Scale

Used for:

```text
router_rms_scale
```

Computation:

```text
O = (A @ B) * r[:, None]
```

Traffic per mapping:

```text
HBM = A_reads
    + W_reads
    + partial_accumulator_reads
    + partial_accumulator_writes
    + final_output_writes
    + row_scale_reads
```

where:

```text
final_output_writes = mt * nt * m0 * n0 * bytes
row_scale_reads = mt * nt * m0 * bytes
buffer = A_tile + W_tile + raw_output_tile + m0 * bytes
```

Correctness reason: RMS scale is row-wise and tile-local once `r` is known.  The
GEMM output tile does not need to be stored and reloaded for scaling; scaling is
applied before final store.  The model still counts the row-scale vector read.

Numerical result at optimum:

```text
router_rms_scale HBM = 152.016 MiB
router_rms_scale OI = 323.361 FLOP/byte
router_rms_scale time = 0.113 ms
```

This does not save much versus the unfused model because `ffn_area.py` did not
materialize a separate normalized hidden tensor before router.

### 2. GEMM + RMS Scale + SwiGLU

Used for:

```text
up_gate_rms_swiglu_x512
```

Computation:

```text
D = (A @ B) * r[:, None]
[G, U] = interleaved_split(D)
O = SiLU(G) * U
```

Let:

```text
N = 2P
p0 = output tile columns after SwiGLU
raw_output_tile = m0 * (2 * p0) * bytes
final_output_tile = m0 * p0 * bytes
```

Traffic per mapping:

```text
HBM = A_reads
    + W_reads
    + partial_accumulator_reads
    + partial_accumulator_writes
    + final_activated_output_writes
    + row_scale_reads
```

where:

```text
W_tile = k0 * (2 * p0) * bytes
partial spill traffic uses raw_output_tile
final_activated_output_writes = mt * pt * m0 * p0 * bytes
row_scale_reads = mt * pt * m0 * bytes
buffer = A_tile + W_tile + raw_output_tile + m0 * bytes
```

Correctness reason: the raw gate/up tile is an accumulator/epilogue-local value.
It is not a final tensor.  The fused epilogue applies RMS scale and SwiGLU before
the output is written, so the raw `2P` gate/up tensor store and reload are
removed.  Partial accumulator spills, if required by loop order, still use the
raw `2P` accumulator tile because the reduction over K must be completed before
SwiGLU is valid.

Numerical result at optimum:

```text
unfused up_gate_x512 + activation traffic = 26624.000 MiB
fused up_gate_rms_swiglu_x512 traffic = 25600.125 MiB
saved traffic = 1023.875 MiB

unfused time = 13.685 ms
fused time = 13.159 ms
speedup = 1.040x
```

### 3. GEMM + Residual + Partial RMS + Weight

This function exists in `coda_fused_traffic.py`, but it is not used in the final
isolated decode-FFN comparison.  It would apply when modeling the previous
projection feeding the FFN boundary:

```text
D = A @ B + C
partial = reduce_tile(D * D)
O = D * gamma
```

Traffic adaptation:

```text
HBM = standard GEMM A/W/partial traffic
    + residual C tile reads
    + gamma vector reads
    + partial RMS-stat writes
    + final O writes
```

Correctness reason: residual add, gamma multiply, and tile-local partial RMS
statistics are all epilogue-local once the GEMM output tile exists.  Only compact
partial statistics need to be written for a later reduction.

### 4. Down + Weighted Sum + Residual

This optimistic fusion was removed from the final model.

Why it is excluded:

```text
out[token, h] = residual[token, h]
              + sum_{i=1..top_k} gate[token, i] * down_i[token, h]
```

For `top_k = 8`, one final output tile depends on multiple expert GEMM output
tiles.  A standard GEMM epilogue only sees one expert's current tile.  Therefore
this requires a specialized grouped/persistent top-k schedule, not just CODA's
standard epilogue abstraction.

The final conservative model keeps:

```text
down_x512
expert_weighted_sum over all 8192 tokens and top_k=8
residual_add
```

as separate stages using the same OI method as `ffn_area.py`.

## Interpretation

### Why Tensor Area Dominates

The total runtime is dominated by tensor-core GEMMs:

```text
unfused up_gate_x512 + down_x512 = 13.290 + 6.842 = 20.133 ms
fused   up_gate_fused + down_x512 = 13.159 + 6.842 = 20.001 ms
```

This is almost the whole end-to-end time.  The vector stages are bandwidth-bound
but small in absolute time:

```text
expert_weighted_sum = 0.444 ms
residual_add = 0.148 ms
RMSNorm square-reduction = 0.049 ms
activation = 0.395 ms in unfused, removed as a separate stage in fused
```

Thus, the optimizer gives nearly all area to tensor cores:

```text
rt ~= 0.98
```

### Why CUDA Area Shrinks With Fusion

In the unfused model, CUDA cores must cover:

```text
RMSNorm square-reduction
activation
expert weighted sum
residual add
```

In the fused model, activation is absorbed into the up-gate epilogue:

```text
RMSNorm square-reduction
expert weighted sum
residual add
small epilogue work inside fused GEMMs
```

Because these stages are mostly memory-bound, a small CUDA allocation is enough
to hit their memory roof.  The optimal `rc` drops:

```text
unfused rc = 0.018
fused rc = 0.014
```

### Why SMEM Stays Small

With `A_total = 136.29 mm^2`, tensor cores are expensive:

```text
A_tensor_core = 150075 um^2
```

The optimizer uses almost all area for tensor cores and only enough SMEM to reach
the useful traffic-frontier knee:

```text
unfused SMEM = 1.128 MiB
fused SMEM = 1.316 MiB
```

Additional SMEM beyond this point does not materially reduce modeled traffic for
the dominant GEMM stages, so it loses to additional tensor-core area.

## Recommendation

For this decode FFN model and these constants:

```text
Use a tensor-core-heavy design.
Allocate only enough CUDA area to saturate low-OI vector/epilogue work.
Allocate SMEM near the traffic-frontier knee, not as a large area fraction.
```

Numerically:

```text
unfused optimum:
  rc = 0.018
  rt = 0.976
  SMEM = 0.006

conservative fused optimum:
  rc = 0.014
  rt = 0.979
  SMEM = 0.007
```

The conservative CODA-style fusion changes area balance only slightly because
the main bottleneck remains tensor-core GEMM throughput.  It improves total time
by about 2.5%, primarily by removing the raw gate/up activation round trip in
the up-gate path.

