"""Sweep the best die-area distribution across decode batch sizes.

For each batch size, reconfigure ``decode_area_latency`` (via its ``configure()``)
and evaluate the full GLM-5.2 decode-layer area grid in-process (no CSV/PNG output),
reporting the area split (rc/rt/r_smem -> CUDA/Tensor/SMEM) that minimizes total layer
time at a fixed KV-cache length.

Default: the "toughest" 1M-token context across batches 512/1024/2048/4096.

    conda run -n profiling python decode_area_sweep.py
    conda run -n profiling python decode_area_sweep.py --seq-len 100000 --batches 256,512
"""

from __future__ import annotations

import argparse

import decode_area_latency as d


def _stage_time(task_times: dict, prefix: str, index: int) -> float:
    key = next((k for k in task_times if k.startswith(prefix)), None)
    return float(task_times[key][index]) if key else 0.0


def sweep(seq_len: int, batches: list[int]) -> None:
    print(f"KV-cache length (seq_len): {seq_len:,} tokens")
    print(
        f"MLA: {d.N_HEADS} heads, kv_lora_rank {d.KV_LORA_RANK}, "
        f"latent/token {d.KV_LATENT} elem; MoE: {d.EXPERTS} experts, top-{d.ROUTER_TOP_K}\n"
    )

    hdr = (
        f"{'batch':>6} {'tok/exp':>7} {'rc':>6} {'rt':>6} {'r_smem':>7} "
        f"{'SMEM MiB':>9} {'CUDA':>5} {'Tensor':>6} {'total s':>10} "
        f"{'attn s':>10} {'attn%':>6} {'HBM GiB':>9} {'TFLOP/s':>8} {'bneck':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for b in batches:
        d.configure(b, seq_len)
        r = d.evaluate_layer()
        i = r["best_index"]
        total = float(r["total_time"][i])
        attn = float(r["attention_time"][i])
        tot_hbm = float(d.total_hbm_traffic_bytes(r["task_traffic"])[i])
        eff = r["modeled_operations"] / total / 1e12
        m = d.attention_core_mapping(
            d.ATTENTION_CORE_TASK,
            float(r["smem_bytes"][i]),
            float(r["tensor_roof"][i]),
            float(r["cuda_roof"][i]),
        )
        print(
            f"{b:>6} {d.TOKENS_PER_EXPERT:>7} {r['rc'][i]:>6.3f} {r['rt'][i]:>6.3f} "
            f"{r['r_smem'][i]:>7.3f} {r['smem_bytes'][i] / 2**20:>9.3f} "
            f"{int(r['cuda_cores'][i]):>5} {int(r['tensor_cores'][i]):>6} "
            f"{total:>10.3f} {attn:>10.3f} {100 * attn / total:>5.1f}% "
            f"{tot_hbm / 2**30:>9.1f} {eff:>8.2f} {m['bottleneck']:>7}"
        )
        rows.append((b, r, i, m))

    print("\nPer-stage time (ms) at each batch's best area node:")
    stage_hdr = (
        f"{'batch':>6} {'attn_mem':>10} {'attn_tensor':>12} {'attn_softmax':>13} "
        f"{'up_gate':>9} {'down':>9} {'mla_o':>8} {'rest':>8}"
    )
    print(stage_hdr)
    for b, r, i, m in rows:
        tt = r["task_times"]
        up = _stage_time(tt, "up_gate", i)
        dn = _stage_time(tt, "down", i)
        mo = _stage_time(tt, "mla_o", i)
        rest = float(r["total_time"][i]) - float(r["attention_time"][i]) - up - dn - mo
        print(
            f"{b:>6} {m['mem_time'] * 1e3:>10.1f} {m['tensor_time'] * 1e3:>12.1f} "
            f"{m['cuda_time'] * 1e3:>13.1f} {up * 1e3:>9.1f} {dn * 1e3:>9.1f} "
            f"{mo * 1e3:>8.2f} {rest * 1e3:>8.2f}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=1_000_000)
    parser.add_argument(
        "--batches",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[512, 1024, 2048, 4096],
        help="comma-separated batch sizes (default 512,1024,2048,4096)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sweep(args.seq_len, args.batches)
