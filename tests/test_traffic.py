from snowcat_demo.model.mapping import enumerate_mappings
from snowcat_demo.model.traffic import (
    LOOP_ORDERS,
    _simulate_mapping,
    estimate_mapping_traffic,
)
from snowcat_demo.model.workload import GemmWorkload


def test_complete_tile_reaches_algorithmic_minimum() -> None:
    workload = GemmWorkload(m=2, k=2, n=2, bytes_per_element=1)

    traffic = estimate_mapping_traffic(workload, 2, 2, 2, ("M", "N", "K"))

    assert traffic.total_bytes == workload.algorithmic_minimum_bytes
    assert traffic.a_read_bytes == 4
    assert traffic.w_read_bytes == 4
    assert traffic.b_read_bytes == 0
    assert traffic.b_write_bytes == 4


def test_k_innermost_avoids_partial_output_reads() -> None:
    workload = GemmWorkload(m=2, k=2, n=2, bytes_per_element=1)

    traffic = estimate_mapping_traffic(workload, 1, 1, 1, ("M", "N", "K"))

    assert traffic.a_read_bytes == 8
    assert traffic.w_read_bytes == 8
    assert traffic.b_read_bytes == 0
    assert traffic.b_write_bytes == 4
    assert traffic.total_bytes == 20


def test_n_innermost_reuses_activation_but_spills_partials() -> None:
    workload = GemmWorkload(m=2, k=2, n=2, bytes_per_element=1)

    traffic = estimate_mapping_traffic(workload, 1, 1, 1, ("M", "K", "N"))

    assert traffic.a_read_bytes == 4
    assert traffic.w_read_bytes == 8
    assert traffic.b_read_bytes == 4
    assert traffic.b_write_bytes == 8
    assert traffic.total_bytes == 24


def test_no_enumerated_mapping_falls_below_algorithmic_minimum() -> None:
    workload = GemmWorkload(m=4, k=4, n=4, bytes_per_element=2)

    for point in enumerate_mappings(workload):
        assert point.backing_store_bytes >= workload.algorithmic_minimum_bytes


def test_enumeration_count_matches_divisors_times_loop_orders() -> None:
    workload = GemmWorkload(m=4, k=8, n=2, bytes_per_element=1)

    points = enumerate_mappings(workload)

    assert len(points) == 3 * 4 * 2 * len(LOOP_ORDERS)


def test_closed_form_matches_tile_simulator_for_small_mapspace() -> None:
    workload = GemmWorkload(m=4, k=4, n=4, bytes_per_element=2)

    for m0 in (1, 2, 4):
        for k0 in (1, 2, 4):
            for n0 in (1, 2, 4):
                for loop_order in LOOP_ORDERS:
                    closed_form = estimate_mapping_traffic(
                        workload, m0, k0, n0, loop_order
                    )
                    simulated, _ = _simulate_mapping(
                        workload,
                        m0,
                        k0,
                        n0,
                        loop_order,
                        capture_trace=False,
                    )
                    assert closed_form == simulated
