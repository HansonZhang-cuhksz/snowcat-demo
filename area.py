from ski_slope import min_attainable_traffic
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Chip Constant
A_total = 694.116 * 10 ** 6             # um^2
A_bit = 0.0864                          # um^2/bit
logic_density = 39.98                   # MTr/mm^2
tensor_logic = 6                        # MTr/tensor
A_tensor = tensor_logic / logic_density * 10 ** 6   # um^2/tensor

tensor_flops = 512 * 1.00 * 10 ** 9     # flops/s/tensor
bw = 2.04 * 10 ** 12                    # byte/s

# GEMM Task
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


def main():
    r = np.arange(0, 1, 0.001)

    sram_bytes = r * A_total / A_bit / 8    # bytes
    tensors = np.floor((1-r) * A_total / A_tensor)    # tensors

    operations = 2 * M * N * K              # flops
    with ProcessPoolExecutor() as executor:
        min_traffic = np.fromiter(
            executor.map(min_traffic_at_capacity, sram_bytes, chunksize=25),
            dtype=float,
            count=len(sram_bytes),
        )                                   # bytes
    oi = operations / min_traffic           # flops/byte
    p_mem = oi * bw                         # flops/s
    p_compute = tensors * tensor_flops      # flops/s
    p_peak = np.minimum(p_mem, p_compute)   # flops/s

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(r, p_peak, label='Peak Performance')
    plt.plot(r, p_mem, label='Memory Performance')
    plt.plot(r, p_compute, label='Compute Performance')
    plt.xlabel('SRAM Utilization')
    plt.ylabel('Performance (FLOPS)')
    plt.title('Performance vs. SRAM Utilization')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./result/p-r.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(r, p_peak, label='Peak Performance')
    plt.xlabel('SRAM Utilization')
    plt.ylabel('Performance (FLOPS)')
    plt.title('Performance vs. SRAM Utilization')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./result/p-r_peak.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(r, p_mem, label='Memory Performance')
    plt.xlabel('SRAM Utilization')
    plt.ylabel('Performance (FLOPS)')
    plt.title('Performance vs. SRAM Utilization')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./result/p-r_mem.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(r, p_compute, label='Compute Performance')
    plt.xlabel('SRAM Utilization')
    plt.ylabel('Performance (FLOPS)')
    plt.title('Performance vs. SRAM Utilization')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./result/p-r_compute.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
