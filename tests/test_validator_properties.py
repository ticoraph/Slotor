"""Property-based tests: incremental == full, reversibility, rigid-motion invariance (D6, D13)."""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from slotor.geometry import Pose, build_track
from slotor.validation import IncrementalValidator, Surface, validate_layout

from .conftest import cached_catalog

SLUGS = ["curve_r1_60", "curve_r2_30", "straight_345", "straight_115"]
SETTINGS = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])

# Curves are over-represented so that random chains often loop back and collide.
steps = st.lists(
    st.tuples(
        st.sampled_from(
            ["curve_r1_60"] * 4 + ["curve_r2_30"] * 2 + ["straight_345", "straight_115"]
        ),
        st.booleans(),
    ),
    min_size=1,
    max_size=24,
)
# Runs of identical pieces: seven R1 60 curves in a row, for instance, revisit the first one.
loopy_steps = st.lists(
    st.tuples(
        st.sampled_from(
            ["curve_r1_60", "curve_r1_60", "curve_r2_30", "straight_345", "straight_115"]
        ),
        st.booleans(),
        st.integers(1, 8),
    ),
    min_size=1,
    max_size=6,
).map(lambda runs: [(slug, flip) for slug, flip, n in runs for _ in range(n)])
inventories = st.fixed_dictionaries({slug: st.integers(0, 12) for slug in SLUGS})
surfaces = st.one_of(
    st.none(),
    st.builds(Surface, st.floats(1500, 4000), st.floats(1500, 3000), st.floats(0, 100)),
)
starts = st.builds(
    Pose,
    st.floats(0, 3000),
    st.floats(0, 2000),
    st.floats(0, 2 * math.pi),
    st.just(0),
)


def _items(raw):
    catalog = cached_catalog()
    return [(catalog.get(slug), flip) for slug, flip in raw]


def _check_equivalence(raw, inventory, surface, start):
    inc = IncrementalValidator(inventory, surface=surface, start=start)
    accepted = []
    for piece, mirrored in _items(raw):
        verdict = inc.check_add(piece, mirrored)
        full = validate_layout(
            [*accepted, (piece, mirrored)],
            inventory,
            surface=surface,
            start=start,
            require_closed=False,
            fail_fast=True,  # same stage order as the incremental path
        )
        assert (not verdict) == full.valid, (verdict, full.issues)
        assert {i.code for i in verdict} == set(full.codes)
        label = (
            "accepted"
            if not verdict
            else "refused: " + ",".join(sorted(c.value for c in full.codes))
        )
        event(label)
        if not verdict:
            assert not inc.add(piece, mirrored)
            accepted.append((piece, mirrored))
    closed_inc = inc.result()
    closed_full = validate_layout(accepted, inventory, surface=surface, start=start)
    assert closed_inc.valid == closed_full.valid
    assert set(closed_inc.codes) == set(closed_full.codes)


@SETTINGS
@given(steps, inventories, surfaces, starts)
def test_incremental_agrees_with_the_full_validator_at_every_step(raw, inventory, surface, start):
    _check_equivalence(raw, inventory, surface, start)


@SETTINGS
@given(loopy_steps, starts)
def test_incremental_agrees_when_collisions_dominate(raw, start):
    """Ample inventory and no surface: refusals are mostly collisions."""
    _check_equivalence(raw, {slug: 99 for slug in SLUGS}, None, start)


@SETTINGS
@given(steps, starts)
def test_add_then_pop_restores_the_state(raw, start):
    inventory = {slug: 99 for slug in SLUGS}
    inc = IncrementalValidator(inventory, start=start)
    for piece, mirrored in _items(raw):
        inc.add(piece, mirrored)
    before = (len(inc), inc.end_pose, dict(inc.used), inc.items)
    probe = cached_catalog().get("straight_345")
    verdict = inc.check_add(probe)
    if not inc.add(probe):
        inc.pop()
    assert (len(inc), inc.end_pose, dict(inc.used), inc.items) == before
    assert inc.check_add(probe) == verdict


@SETTINGS
@given(steps, starts, st.floats(0, 2 * math.pi))
def test_collision_verdict_is_invariant_under_rigid_motion(raw, start, angle):
    inventory = {slug: 99 for slug in SLUGS}
    items = _items(raw)
    base = validate_layout(items, inventory, start=start, require_closed=False, fail_fast=False)
    moved_start = Pose(start.x + 321.0, start.y - 123.0, start.theta + angle, 0)
    moved = validate_layout(
        items, inventory, start=moved_start, require_closed=False, fail_fast=False
    )
    assert base.codes == moved.codes


@SETTINGS
@given(st.data())
def test_a_valid_loop_is_really_closed_and_within_inventory(data):
    """Whatever the validator accepts as a loop satisfies independent invariants."""
    raw = data.draw(steps)
    inventory = {slug: 99 for slug in SLUGS}
    items = _items(raw)
    result = validate_layout(items, inventory)
    if result.valid:
        track = build_track(items)
        assert track.total_turn_deg % 360.0 == pytest.approx(0.0, abs=1e-6) or (
            abs(track.total_turn_deg % 360.0 - 360.0) < 1e-6
        )
        assert track.closure_error().distance_mm <= 10.0 + 1e-9
        assert result.score_mm == pytest.approx(track.length_mm)
    else:
        assert result.score_mm == 0.0


@SETTINGS
@given(st.integers(0, 9), st.integers(1, 3))
def test_oval_with_pieces_removed_is_never_accepted(first, count):
    catalog = cached_catalog()
    curve, straight = catalog.get("curve_r1_60"), catalog.get("straight_345")
    items = [(curve, False)] * 3 + [(straight, False)] * 2 + [(curve, False)] * 3
    items += [(straight, False)] * 2
    damaged = items[:first] + items[first + count :]
    result = validate_layout(damaged, {"curve_r1_60": 6, "straight_345": 4})
    assert not result.valid
    assert result.score_mm == 0.0


def _random_closed_candidates(rng, count):
    """Chains that often close: stadiums with random straights, sometimes perturbed."""
    catalog = cached_catalog()
    curve = catalog.get("curve_r1_60")
    straights = [catalog.get("straight_345"), catalog.get("straight_115")]
    for _ in range(count):
        flip = rng.random() < 0.5
        side_a = [rng.choice(straights) for _ in range(rng.randint(0, 5))]
        side_b = list(side_a) if rng.random() < 0.8 else [rng.choice(straights) for _ in side_a]
        items = [(curve, flip)] * 3 + [(p, False) for p in side_a]
        items += [(curve, flip)] * 3 + [(p, False) for p in side_b]
        if rng.random() < 0.3:
            index = rng.randrange(len(items) + 1)
            items.insert(index, (rng.choice([curve, *straights]), rng.random() < 0.5))
        if rng.random() < 0.2 and items:
            items.pop(rng.randrange(len(items)))
        yield items


def test_no_invalid_loop_is_accepted_among_10000_random_candidates():
    """Independent oracle: an accepted loop closes, respects the inventory and barely overlaps."""
    import random

    rng = random.Random(20260511)
    inventory = {slug: 99 for slug in SLUGS}
    accepted = 0
    for items in _random_closed_candidates(rng, 10_000):
        result = validate_layout(items, inventory)
        if not result.valid:
            assert result.score_mm == 0.0
            continue
        accepted += 1
        track = build_track(items)
        assert track.closure_error().distance_mm <= 10.0 + 1e-9
        assert round(track.total_turn_deg) % 360 == 0
        shapes = [p.outline for p in track.pieces]
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                assert shapes[i].intersection(shapes[j]).area < 1000.0, (i, j)
    assert accepted > 1000, (
        "the generator should produce many valid loops, or the test proves nothing"
    )
