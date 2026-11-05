"""Seeds and Beam Search (Phase 3): validity, closure, determinism, admissibility of the bound."""

from __future__ import annotations

import functools
import json

import pytest

from slotor.geometry import Pose, ToleranceConfig, local_geometry
from slotor.optimizers import (
    BeamConfig,
    Problem,
    beam_search,
    can_still_close,
    finalize,
    generate_seeds,
)
from slotor.validation import IncrementalValidator, Surface, validate_layout

from .conftest import FIXTURES, cached_catalog

TOL = ToleranceConfig()
INSTANCES = json.loads((FIXTURES / "optim_instances.json").read_text(encoding="utf-8"))["instances"]
NAMES = [i["name"] for i in INSTANCES]
FAST = BeamConfig(width=60, max_expansions=15_000)


def problem(name: str) -> Problem:
    spec = next(i for i in INSTANCES if i["name"] == name)
    width, height = spec["surface_mm"]
    return Problem(cached_catalog(), spec["inventory"], Surface(width, height))


def assert_valid_closed(prob: Problem, solution) -> None:
    result = validate_layout(
        solution.items, prob.inventory, surface=prob.surface, start=solution.start, tol=prob.tol
    )
    assert result.valid, result.issues
    assert result.track is not None and result.track.is_closed(prob.tol)
    assert solution.length_mm == pytest.approx(result.track.length_mm)


def test_unmodelled_pieces_are_ignored_never_used():
    prob = problem("golden_a")
    assert prob.ignored == {
        "control_unit": 1,
        "lane_change_double_1": 2,
        "lane_change_double_2": 2,
    }
    solution = beam_search(prob, seed=0, config=FAST).best
    assert solution is not None
    assert not {slug for slug, _ in solution.slugs} & set(prob.ignored)


@pytest.mark.parametrize("name", NAMES)
def test_seed_is_a_valid_closed_loop_within_the_inventory(name):
    prob = problem(name)
    seeds = generate_seeds(prob)
    assert seeds, "every instance admits at least one stadium"
    for seed in seeds[:3]:
        assert_valid_closed(prob, seed)
    assert [s.length_mm for s in seeds] == sorted((s.length_mm for s in seeds), reverse=True)


def test_oval_kit_seed_is_the_control_oval():
    seed = generate_seeds(problem("kit_oval"))[0]
    assert seed.length_mm == pytest.approx(6 * 297.0 * 3.141592653589793 / 3 + 4 * 345.0)
    assert seed.n_pieces == 10


def test_stadium_sides_may_be_composed_differently():
    prob = Problem(
        cached_catalog(),
        {"curve_r1_60": 6, "straight_345": 1, "straight_115": 3},
        Surface(3000, 2000),
    )
    best = generate_seeds(prob)[0]
    assert_valid_closed(prob, best)
    # 345 on one side, 3 x 115 on the other: both sides measure 345 mm.
    slugs = [slug for slug, _ in best.slugs]
    assert slugs.count("straight_345") == 1 and slugs.count("straight_115") == 3
    assert best.length_mm == pytest.approx(6 * 297.0 * 3.141592653589793 / 3 + 2 * 345.0)


@pytest.mark.parametrize("name", NAMES)
def test_beam_returns_a_valid_closed_circuit_no_worse_than_its_seed(name):
    prob = problem(name)
    seed_length = generate_seeds(prob)[0].length_mm
    result = beam_search(prob, seed=3, config=FAST)
    assert result.best is not None
    assert_valid_closed(prob, result.best)
    assert result.best.length_mm >= seed_length - 1e-9


def test_beam_improves_on_the_seed_of_a_roomy_instance():
    prob = problem("golden_b")
    seed_length = generate_seeds(prob)[0].length_mm
    result = beam_search(prob, seed=0, config=BeamConfig(width=150, max_expansions=40_000))
    assert result.best is not None and result.best.length_mm > seed_length + 100.0


@pytest.mark.parametrize("name", ["kit_mixed", "golden_b"])
def test_same_seed_same_result(name):
    prob = problem(name)
    first = beam_search(prob, seed=11, config=FAST)
    second = beam_search(prob, seed=11, config=FAST)
    assert first.best is not None and second.best is not None
    assert first.best.slugs == second.best.slugs
    assert first.best.start == second.best.start
    assert first.expansions == second.expansions


def test_history_is_anytime_and_strictly_improving():
    result = beam_search(problem("golden_b"), seed=2, config=FAST)
    lengths = [h.length_mm for h in result.history]
    assert lengths == sorted(set(lengths))
    times = [h.elapsed_s for h in result.history]
    assert times == sorted(times)
    assert result.best is not None and lengths[-1] == pytest.approx(result.best.length_mm)


def test_expansion_budget_is_respected():
    result = beam_search(
        problem("golden_b"), seed=0, config=BeamConfig(width=60, max_expansions=500)
    )
    assert result.stop_reason == "expansions"
    assert 500 <= result.expansions < 500 + 40  # finishes the node in progress at most


def test_time_budget_stops_the_search():
    config = BeamConfig(width=2000, time_budget_s=0.2)
    result = beam_search(problem("golden_b"), seed=0, config=config)
    assert result.stop_reason in {"time", "exhausted"}
    assert result.elapsed_s < 3.0


def test_no_solution_when_the_inventory_cannot_close():
    prob = Problem(cached_catalog(), {"curve_r1_60": 2, "straight_345": 3}, Surface(3000, 2000))
    result = beam_search(prob, seed=0, config=FAST)
    assert result.best is None
    assert generate_seeds(prob) == []


def test_no_solution_when_the_surface_is_too_small():
    prob = Problem(cached_catalog(), {"curve_r1_60": 6, "straight_345": 4}, Surface(780, 780))
    assert beam_search(prob, seed=0, config=FAST).best is None


def test_finalize_refuses_a_layout_that_does_not_fit():
    prob = problem("kit_oval")
    items = generate_seeds(prob)[0].items
    small = Problem(prob.catalog, prob.inventory, Surface(1400, 700))
    assert finalize(small, items) is None


def test_finalize_turns_a_layout_to_fit_the_transposed_surface():
    prob = problem("kit_oval")
    items = generate_seeds(prob)[0].items
    tall = Problem(prob.catalog, prob.inventory, Surface(800, 1500))
    solution = finalize(tall, items)
    assert solution is not None
    assert solution.start.theta == pytest.approx(3.141592653589793 / 2)
    assert_valid_closed(tall, solution)


# --- admissibility of the closure bound against exhaustive enumeration ---------------------------


def _enumerate_closed(inventory: dict[str, int]):
    return _enumerate_closed_cached(tuple(sorted(inventory.items())))


@functools.cache
def _enumerate_closed_cached(items: tuple[tuple[str, int], ...]):
    """Every collision-free closed sequence (free space), found without any bound."""
    inventory = dict(items)
    catalog = cached_catalog()
    pieces = [catalog.get(slug) for slug in sorted(inventory)]
    closed: list[tuple[tuple[tuple[str, bool], ...], float]] = []
    validator = IncrementalValidator(inventory, start=Pose(0.0, 0.0, 0.0, 0), tol=TOL)

    def recurse() -> None:
        for piece in pieces:
            for mirrored in (False,) if piece.category == "straight" else (False, True):
                if validator.add(piece, mirrored):
                    continue
                if validator.result().valid:
                    track = validator.track()
                    key = tuple((p.slug, m) for p, m in validator.items)
                    closed.append((key, track.length_mm))
                else:
                    recurse()
                validator.pop()

    recurse()
    return closed


@pytest.mark.parametrize(
    "inventory",
    [
        {"curve_r1_60": 6, "straight_345": 2, "straight_115": 1},
        {"curve_r1_60": 4, "curve_r1_30": 4},
    ],
)
def test_closure_bound_never_cuts_a_branch_that_closes(inventory):
    catalog = cached_catalog()
    closed = _enumerate_closed(inventory)
    assert closed, "the inventory should admit closed loops"
    for sequence, _ in closed:
        validator = IncrementalValidator(inventory, start=Pose(0.0, 0.0, 0.0, 0), tol=TOL)
        remaining_length = sum(
            local_geometry(catalog.get(s), False, TOL).length_mm * n for s, n in inventory.items()
        )
        remaining_turn = sum(
            (catalog.get(s).turn_angle_deg or 0.0) * n for s, n in inventory.items()
        )
        for slug, mirrored in sequence[:-1]:
            piece = catalog.get(slug)
            assert not validator.add(piece, mirrored)
            remaining_length -= local_geometry(piece, False, TOL).length_mm
            remaining_turn -= piece.turn_angle_deg or 0.0
            assert can_still_close(
                validator.end_pose,
                Pose(0.0, 0.0, 0.0, 0),
                remaining_length,
                remaining_turn,
                TOL,
                198.0,
            ), (sequence, len(validator))


def test_beam_matches_the_exhaustive_optimum_on_a_tiny_inventory():
    inventory = {"curve_r1_60": 6, "straight_345": 2, "straight_115": 1}
    optimum = max(length for _, length in _enumerate_closed(inventory))
    prob = Problem(cached_catalog(), inventory, Surface(3000, 3000))
    result = beam_search(prob, seed=0, config=BeamConfig(width=400, use_seeds=False))
    assert result.best is not None
    assert result.best.length_mm == pytest.approx(optimum)
