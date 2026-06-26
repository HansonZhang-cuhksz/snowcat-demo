from snowcat_demo.model.decision import (
    Bottleneck,
    Investment,
    compare_next_area_increment,
    doubling_buffer_gain_percent,
    evaluate_capacity,
)
from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.workload import GemmWorkload


def test_evaluate_capacity_reports_memory_bottleneck_when_bandwidth_limit_is_lower() -> None:
    workload = GemmWorkload(16, 16, 16, 2)
    points = enumerate_mappings(workload)

    result = evaluate_capacity(
        workload,
        points,
        capacity_bytes=512,
        memory_bandwidth_gb_s=1.0,
        peak_compute_tflops=100.0,
    )

    assert result is not None
    assert result.bottleneck == Bottleneck.MEMORY
    assert result.performance_tflops < 100.0


def test_evaluate_capacity_reports_compute_bottleneck_when_peak_is_lower() -> None:
    workload = GemmWorkload(16, 16, 16, 2)
    points = enumerate_mappings(workload)

    result = evaluate_capacity(
        workload,
        points,
        capacity_bytes=4096,
        memory_bandwidth_gb_s=100_000.0,
        peak_compute_tflops=1.0,
    )

    assert result is not None
    assert result.bottleneck == Bottleneck.COMPUTE
    assert result.performance_tflops == 1.0


def test_next_area_recommends_sram_when_sram_improves_performance_more() -> None:
    workload = GemmWorkload(64, 64, 64, 2)
    points = enumerate_mappings(workload)

    decision = compare_next_area_increment(
        workload,
        points,
        capacity_bytes=1024,
        memory_bandwidth_gb_s=100.0,
        peak_compute_tflops=1_000.0,
        sram_increment_bytes=64 * 1024,
        compute_increment_tflops=100.0,
    )

    assert decision.recommendation == Investment.SRAM
    assert decision.sram_gain_percent > decision.compute_gain_percent


def test_next_area_recommends_compute_when_compute_improves_performance_more() -> None:
    workload = GemmWorkload(16, 16, 16, 2)
    points = enumerate_mappings(workload)

    decision = compare_next_area_increment(
        workload,
        points,
        capacity_bytes=4096,
        memory_bandwidth_gb_s=100_000.0,
        peak_compute_tflops=1.0,
        sram_increment_bytes=64 * 1024,
        compute_increment_tflops=1.0,
    )

    assert decision.recommendation == Investment.COMPUTE
    assert decision.compute_gain_percent > decision.sram_gain_percent


def test_doubling_buffer_gain_is_non_negative_when_mapping_exists() -> None:
    workload = GemmWorkload(32, 32, 32, 2)
    points = enumerate_mappings(workload)

    gain = doubling_buffer_gain_percent(
        workload,
        points,
        capacity_bytes=1024,
        memory_bandwidth_gb_s=100.0,
        peak_compute_tflops=100.0,
    )

    assert gain is not None
    assert gain >= 0.0

