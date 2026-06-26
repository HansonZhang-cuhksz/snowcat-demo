from snowcat_demo.model.mapping import GemmMapping, MappingPoint
from snowcat_demo.model.pareto import best_at_capacity, is_pareto_point, pareto_frontier
from snowcat_demo.model.traffic import TrafficBreakdown


def make_point(buffer: int, traffic: int, m0: int = 1) -> MappingPoint:
    return MappingPoint(
        mapping=GemmMapping(m0=m0, k0=1, n0=1, loop_order=("M", "N", "K")),
        traffic=TrafficBreakdown(
            buffer_bytes=buffer,
            a_read_bytes=traffic,
            w_read_bytes=0,
            b_read_bytes=0,
            b_write_bytes=0,
        ),
    )


def test_pareto_frontier_removes_dominated_points() -> None:
    points = [
        make_point(10, 100, 1),
        make_point(20, 80, 2),
        make_point(20, 90, 3),
        make_point(30, 80, 4),
        make_point(40, 50, 5),
    ]

    frontier = pareto_frontier(points)

    assert [(p.buffer_bytes, p.backing_store_bytes) for p in frontier] == [
        (10, 100),
        (20, 80),
        (40, 50),
    ]
    assert not is_pareto_point(points[2], frontier)
    assert not is_pareto_point(points[3], frontier)


def test_frontier_traffic_is_monotonic_non_increasing() -> None:
    points = [make_point(10, 100, 1), make_point(20, 70, 2), make_point(30, 60, 3)]
    frontier = pareto_frontier(points)
    traffic = [point.backing_store_bytes for point in frontier]

    assert traffic == sorted(traffic, reverse=True)


def test_best_at_capacity_selects_lowest_traffic_eligible_mapping() -> None:
    points = [make_point(10, 100, 1), make_point(20, 70, 2), make_point(40, 50, 3)]

    best = best_at_capacity(points, 25)

    assert best is not None
    assert best.buffer_bytes == 20
    assert best.backing_store_bytes == 70


def test_best_at_capacity_returns_none_when_capacity_too_small() -> None:
    assert best_at_capacity([make_point(10, 100)], 5) is None

