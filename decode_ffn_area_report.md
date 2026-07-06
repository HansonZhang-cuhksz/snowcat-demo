# Decode FFN Area-Balance Report

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
11. HBM Latency ≈ A100
    - 500 cycles
12. SwiGLU FLOPS per element: 8

<!-- ## Formulas

```text
top_k = experts * tokens_per_expert / batch_tokens = 512 * 128 / 8192 = 8

rc = CUDA-core area fraction
rt = tensor-core area fraction
r_smem = 1 - rc - rt

A_cuda_core = 0.2e6 / 39.98 = 5002.501 um^2
A_tensor_core = 6.0e6 / 39.98 = 150075.038 um^2

cuda_cores = floor(rc * A_total / A_cuda_core)
tensor_cores = floor(rt * A_total / A_tensor_core)
smem_bytes = r_smem * A_total / A_bit / 8

cuda_roof = cuda_cores * 5.64e9
tensor_roof = tensor_cores * 512e9
```

```text
Standard GEMM:
ops = 2 * M * N * K
OI = ops / min_HBM_traffic
time = ops / min(tensor_roof, bw * OI)

Vector/reduction:
OI = ops / HBM_traffic
time = ops / min(cuda_roof, bw * OI)

Fused GEMM epilogue:
time = max(tensor_ops / tensor_roof,
           cuda_epilogue_ops / cuda_roof,
           fused_HBM_traffic / bw)

Latency-aware HBM:
latency_seconds = 500 / 1.410e9 = 354.61 ns
required_smem = num_stages * stage_bytes
BW_eff = min(bw, num_stages * stage_bytes / latency_seconds)
``` -->

## Workloads

| Stage | Unfused | Fused |
|---|---|---|
| RMSNorm | square-reduction | square-reduction |
| Router | router GEMM | router + RMS-scale |
| Up/Gate | up_gate GEMM x512 + SwiGLU | up_gate + RMS-scale + SwiGLU x512 |
| Down | down GEMM x512 | down GEMM x512 |
| Expert combine | weighted sum over 8192 tokens, top-k=8 | same |
| Output | residual add | same |

| GEMM | Shape | Count | Ops |
|---|---:|---:|---:|
| router | M=8192, N=512, K=6144 | 1 | 51.540 GFLOP |
| up_gate | M=128, N=4096, K=6144 | 512 | 3298.535 GFLOP |
| down | M=128, N=6144, K=2048 | 512 | 1649.267 GFLOP |

| Vector/reduction | Ops | HBM traffic | OI |
|---|---:|---:|---:|
| RMSNorm square-reduction | 100.655 MFLOP | 96.031 MiB | 0.9996 |
| activation | 1073.742 MFLOP | 768.000 MiB | 1.3333 |
| expert weighted sum | 754.975 MFLOP | 864.125 MiB | 0.8332 |
| residual add | 50.332 MFLOP | 288.000 MiB | 0.1667 |

## Graphs

| Original roofline | Latency-aware roofline |
|---|---|
| ![Unfused original](img/ffn_area_register_accumulator_total_time.png) | ![Unfused latency](img/ffn_area_latency_register_accumulator_total_time.png) |
| ![Fused original](img/ffn_fused_area_total_time.png) | ![Fused latency](img/ffn_fused_area_latency_total_time.png) |

## Area Results

| Model | Workload | rc | rt | SMEM frac | SMEM MiB | CUDA cores | Tensor cores | Time | Throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | Unfused | 0.018 | 0.976 | 0.006 | 1.128 | 490 | 886 | 21.283 ms | 234.996 TFLOP/s |
| Original | Fused | 0.014 | 0.979 | 0.007 | 1.316 | 381 | 889 | 20.756 ms | 240.972 TFLOP/s |
| Latency-aware | Unfused | 0.018 | 0.960 | 0.022 | 4.137 | 490 | 871 | 21.285 ms | 234.974 TFLOP/s |
| Latency-aware | Fused | 0.014 | 0.964 | 0.022 | 4.137 | 381 | 875 | 20.758 ms | 240.951 TFLOP/s |

| Comparison | Time ratio | Throughput ratio | SMEM MiB ratio |
|---|---:|---:|---:|
| Fused vs. unfused, original | 1.0254x faster | 1.0254x higher | 1.167x |
| Fused vs. unfused, latency-aware | 1.0254x faster | 1.0254x higher | 1.000x |
| Latency-aware vs. original, unfused | 1.00009x slower | 0.99991x | 3.667x |
| Latency-aware vs. original, fused | 1.00009x slower | 0.99991x | 3.143x |

## Stage Results

| Stage/group | Unfused time | Fused time | Unfused HBM | Fused HBM | Unfused OI | Fused OI |
|---|---:|---:|---:|---:|---:|---:|
| RMSNorm square-reduction | 0.049 ms | 0.049 ms | 96.031 MiB | 96.031 MiB | 1.000 | 1.000 |
| router / router_rms_scale | 0.114 ms | 0.113 ms | 152.000 MiB | 152.016 MiB | 323.368 | 323.361 |
| up_gate + activation / up_gate_rms_swiglu | 13.685 ms | 13.159 ms | 26624.000 MiB | 25600.125 MiB | 118.192 | 122.929 |
| down_x512 | 6.842 ms | 6.842 ms | 13312.000 MiB | 13312.000 MiB | 118.154 | 118.154 |
| expert_weighted_sum | 0.444 ms | 0.444 ms | 864.125 MiB | 864.125 MiB | 0.833 | 0.833 |
| residual_add | 0.148 ms | 0.148 ms | 288.000 MiB | 288.000 MiB | 0.167 | 0.167 |

| Fused stage/group | Saved HBM | Time saved | Speedup |
|---|---:|---:|---:|
| router_rms_scale | -0.016 MiB | 0.000 ms | 1.003x |
| up_gate_rms_swiglu | 1023.875 MiB | 0.526 ms | 1.040x |
| down + weighted sum + residual | 0.000 MiB | 0.000 ms | 1.000x |

## Latency-Aware Stage Results

| Stage/group | Unfused time | Fused time | Unfused HBM | Fused HBM | Unfused OI | Fused OI |
|---|---:|---:|---:|---:|---:|---:|
| RMSNorm square-reduction | 0.049 ms | 0.049 ms | 96.031 MiB | 96.031 MiB | 1.000 | 1.000 |
| router / router_rms_scale | 0.116 ms | 0.115 ms | 152.000 MiB | 152.016 MiB | 323.368 | 323.361 |
| up_gate + activation / up_gate_rms_swiglu | 13.685 ms | 13.159 ms | 26624.000 MiB | 25600.125 MiB | 118.192 | 122.929 |
| down_x512 | 6.842 ms | 6.842 ms | 13312.000 MiB | 13312.000 MiB | 118.154 | 118.154 |
| expert_weighted_sum | 0.444 ms | 0.444 ms | 864.125 MiB | 864.125 MiB | 0.833 | 0.833 |
| residual_add | 0.148 ms | 0.148 ms | 288.000 MiB | 288.000 MiB | 0.167 | 0.167 |
