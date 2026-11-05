"""LNS (Phase 4): improvement over the seeds, validity of every accepted state, reproducibility."""

from __future__ import annotations

import random

import pytest

from slotor.geometry import local_geometry
from slotor.optimizers import (
    BeamConfig,
    LNSConfig,
    beam_search,
    generate_seeds,
    lns,
    make_solution,
)
from slotor.optimizers.beam import Budget, run_beam
from slotor.optimizers.lns import _repair
from slotor.validation import IncrementalValidator

from .test_optimizers import INSTANCES, TOL, assert_valid_closed, problem

IMPROVABLE = [i["name"] for i in INSTANCES if i["name"] != "kit_oval"]
QUICK = LNSConfig(max_iterations=30)


@pytest.mark.parametrize("name", IMPROVABLE)
def test_lns_improves_on_the_seed_and_stays_valid(name):
    prob = problem(name)
    seed_length = generate_seeds(prob)[0].length_mm
    result = lns(prob, seed=0, config=QUICK)
    assert result.best is not None
    assert_valid_closed(prob, result.best)
    assert result.best.length_mm > seed_length + 50.0, (name, seed_length, result.best.length_mm)


def test_lns_cannot_improve_a_loop_that_already_uses_the_whole_inventory():
    prob = problem("kit_oval")
    result = lns(prob, seed=0, config=QUICK)
    assert result.best is not None
    assert result.best.length_mm == pytest.approx(generate_seeds(prob)[0].length_mm)
    assert result.stats.accepted == 0


def test_same_seed_same_result():
    prob = problem("golden_b")
    first = lns(prob, seed=5, config=QUICK)
    second = lns(prob, seed=5, config=QUICK)
    assert first.best is not None and second.best is not None
    assert first.best.slugs == second.best.slugs
    assert first.best.start == second.best.start
    assert first.stats.attempts == second.stats.attempts


def test_iteration_budget_is_respected():
    result = lns(problem("golden_b"), seed=1, config=QUICK)
    assert result.iterations == 30 and sum(result.stats.attempts.values()) == 30
    assert result.stop_reason == "iterations"


def test_history_is_monotone_and_ends_at_the_best():
    result = lns(problem("kit_mixed"), seed=0, config=QUICK)
    lengths = [h.length_mm for h in result.history]
    assert lengths == sorted(set(lengths))
    assert result.best is not None and lengths[-1] == pytest.approx(result.best.length_mm)
    assert [h.elapsed_s for h in result.history] == sorted(h.elapsed_s for h in result.history)


def test_lns_can_continue_from_a_beam_solution_and_never_gets_worse():
    prob = problem("kit_mixed")
    beam = beam_search(prob, seed=0, config=BeamConfig(width=40, max_expansions=4000)).best
    assert beam is not None
    result = lns(prob, seed=0, config=LNSConfig(max_iterations=15), initial=beam)
    assert result.best is not None
    assert result.best.length_mm >= beam.length_mm - 1e-9
    assert_valid_closed(prob, result.best)


def test_time_budget_stops_the_search():
    config = LNSConfig(time_budget_s=0.3)
    result = lns(problem("golden_a"), seed=0, config=config)
    assert result.stop_reason == "time"
    assert result.elapsed_s < 3.0


def test_restarts_keep_the_best_and_stay_valid():
    prob = problem("golden_b")
    config = LNSConfig(max_iterations=40, stagnation_limit=6)
    result = lns(prob, seed=0, config=config)
    assert result.stats.restarts >= 2
    assert result.best is not None
    assert_valid_closed(prob, result.best)
    assert result.best.length_mm >= generate_seeds(prob)[0].length_mm


def test_no_start_when_no_seed_and_no_initial():
    from slotor.optimizers import Problem
    from slotor.validation import Surface

    from .conftest import cached_catalog

    prob = Problem(cached_catalog(), {"curve_r1_60": 2, "straight_345": 3}, Surface(3000, 2000))
    result = lns(prob, seed=0, config=QUICK)
    assert result.best is None and result.stop_reason == "no_start"


def test_config_validation():
    with pytest.raises(ValueError):
        LNSConfig(k_min=5, k_start=3)
    with pytest.raises(ValueError):
        LNSConfig(weights=(("teleport", 1),))


# --- the repair machinery itself ---------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 6])
def test_repair_can_always_rebuild_a_removed_window(k):
    """Destroying a window of a valid loop leaves a gap the repair must be able to close again."""
    prob = problem("kit_mixed")
    seed = generate_seeds(prob)[0]
    n = seed.n_pieces
    for start_index in (0, n // 2, n - 2):  # includes a window that wraps around the loop end
        track = seed.validation.track
        rotated = seed.items[start_index:] + seed.items[:start_index]
        placed = track.pieces[start_index:] + track.pieces[:start_index]
        target = placed[k].entry
        root = IncrementalValidator(prob.inventory, surface=prob.surface, start=target, tol=TOL)
        for piece, mirrored in rotated[k:]:
            assert not root.add(piece, mirrored)
        found = []

        def accept(items, length, found=found, target=target):
            solution = make_solution(prob, items, target)
            assert solution is not None, "a closed candidate must be valid"
            found.append(solution)
            return length

        run_beam(
            prob,
            root,
            target,
            config=BeamConfig(width=200),
            rng=random.Random(0),
            budget=Budget(None, 40_000),
            incumbent=0.0,
            accept=accept,
            inner=None,
            strict=False,
        )
        assert found, f"no repair found for window {start_index}+{k}"
        assert max(s.length_mm for s in found) >= seed.length_mm - 1e-6
        for solution in found:
            assert_valid_closed(prob, solution)


def test_repair_returns_none_when_nothing_better_exists():
    prob = problem("kit_oval")
    seed = generate_seeds(prob)[0]
    cfg = LNSConfig(repair_expansions=2000)
    for index in range(seed.n_pieces):
        assert (
            _repair(
                prob,
                seed,
                index,
                3,
                config=cfg,
                rng=random.Random(index),
                budget=Budget(None, None),
                floor=None,
            )
            is None
        )


def test_destroyed_piece_lengths_are_accounted_for():
    prob = problem("kit_mixed")
    seed = generate_seeds(prob)[0]
    total = sum(local_geometry(p, m, TOL).length_mm for p, m in seed.items)
    assert total == pytest.approx(seed.length_mm)


def test_adaptive_lns_is_valid_deterministic_and_improves():
    prob = problem("golden_b")
    config = LNSConfig(max_iterations=30, adaptive=True)
    first = lns(prob, seed=3, config=config)
    second = lns(prob, seed=3, config=config)
    assert first.best is not None and second.best is not None
    assert first.best.slugs == second.best.slugs
    assert_valid_closed(prob, first.best)
    assert first.best.length_mm > generate_seeds(prob)[0].length_mm
    assert sum(first.stats.attempts.values()) == 30
