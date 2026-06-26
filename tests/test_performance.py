from snowcat_demo.model.mapping import GemmMapping, MappingPoint
from snowcat_demo.model.performance import (
    attainable_metrics,
    operational_intensity,
    throughput_tflops,
)
from snowcat_demo.model.traffic import TrafficBreakdown
from snowcat_demo.model.workload import GemmWorkload


def make_point(buffer: int, traffic: int) -> MappingPoint:
    return MappingPoint(
        mapping=GemmMapping(m0=buffer, k0=1, n0=1, loop_order=("M", "N", "K")),
        traffic=TrafficBreakdown(
            buffer_bytes=buffer,
            a_read_bytes=traffic,
            w_read_bytes=0,
            b_read_bytes=0,
            b_write_bytes=0,
        ),
    )


def test_operational_intensity_is_ops_per_byte() -> None:
    workload = GemmWorkload(m=4, k=4, n=4, bytes_per_element=1)

    assert operational_intensity(workload, 64) == 2.0


def test_throughput_is_min_of_compute_and_bandwidth_limit() -> None:
    assert throughput_tflops(10.0, memory_bandwidth_gb_s=100.0, peak_compute_tflops=5.0) == 1.0
    assert throughput_tflops(100.0, memory_bandwidth_gb_s=100.0, peak_compute_tflops=5.0) == 5.0


def test_attainable_metrics_follow_frontier_capacities() -> None:
    workload = GemmWorkload(m=4, k=4, n=4, bytes_per_element=1)
    points = [make_point(10, 100), make_point(20, 80), make_point(30, 90)]

    metrics = attainable_metrics(
        workload,
        points,
        memory_bandwidth_gb_s=100.0,
        peak_compute_tflops=10.0,
    )

    assert [metric.capacity_bytes for metric in metrics] == [10, 20]
    assert [metric.traffic_bytes for metric in metrics] == [100, 80]
    assert metrics[1].operational_intensity > metrics[0].operational_intensity

