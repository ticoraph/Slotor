"""Bridges (D30): three elements up, one or more on the deck, three down, on standard pieces."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from slotor.catalog.bridges import DECK_LEVEL, bridge_variants, variants_of
from slotor.catalog.model import ELEVATED_KEY
from slotor.geometry import Pose, ToleranceConfig, local_geometry
from slotor.optimizers import (
    BeamConfig,
    LNSConfig,
    Problem,
    beam_search,
    generate_seeds,
    lns,
    make_solution,
)
from slotor.report import (
    bill_of_materials,
    build_report,
    format_summary,
    solution_from_json,
    solution_to_json,
    solution_to_svg,
)
from slotor.service import OptimizeRequest, build_problem
from slotor.validation import IncrementalValidator, IssueCode, Surface, validate_layout

from .conftest import FIXTURES, cached_catalog
from .test_optimizers import assert_valid_closed

TOL = ToleranceConfig()
STEPS = ["@u0", "@u1", "@u2", "@deck", "@d2", "@d1", "@d0"]


def fixture():
    return json.loads((FIXTURES / "bridge_loop.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bridge_problem():
    data = fixture()
    width, height = data["surface_mm"]
    return Problem(cached_catalog(), data["inventory"], Surface(width, height), bridges=True)


def golden_items(problem: Problem, shift: int = 0, replace: dict | None = None):
    """The fixture loop; ``shift`` moves the whole bridge along the loop, ``replace`` edits it."""
    sequence = [(slug, mirrored) for slug, mirrored in fixture()["sequence"]]
    if shift:
        bases = [slug.split("@")[0] for slug, _ in sequence]
        tags = {i: slug.partition("@")[1:] for i, (slug, _) in enumerate(sequence) if "@" in slug}
        sequence = [
            (bases[i] + "".join(tags.get(i - shift, ("", ""))), m)
            for i, (_, m) in enumerate(sequence)
        ]
    if replace:
        sequence = [(replace.get(i, slug), m) for i, (slug, m) in enumerate(sequence)]
    return [(problem.piece(slug), mirrored) for slug, mirrored in sequence]


def golden_start() -> Pose:
    s = fixture()["start"]
    return Pose(s["x"], s["y"], math.radians(s["theta_deg"]), 0)


def check(problem, items, **kwargs):
    return validate_layout(
        items,
        problem.inventory,
        surface=problem.surface,
        start=golden_start(),
        tol=TOL,
        **kwargs,
    )


# --- the variants are the standard pieces on a staircase of levels ----------------------------


def test_each_raisable_piece_has_the_seven_level_variants():
    catalog = cached_catalog()
    variants = bridge_variants(catalog)
    for slug in ["straight_345"] + [
        p.slug for p in catalog.pieces if p.category == "curve" and p.verified
    ]:
        for step in STEPS:
            assert slug + step in variants, slug + step
    levels = [
        (variants["straight_345" + s].level_in, variants["straight_345" + s].level_out)
        for s in STEPS
    ]
    assert levels == [(0, 1), (1, 2), (2, 3), (3, 3), (3, 2), (2, 1), (1, 0)]
    assert DECK_LEVEL == 3


def test_short_straights_never_climb_or_carry_a_deck():
    variants = bridge_variants(cached_catalog())
    assert not any(slug.startswith(("straight_115", "straight_86_25")) for slug in variants)


def test_variants_have_exactly_the_geometry_and_stock_of_their_base():
    catalog = cached_catalog()
    for slug, variant in bridge_variants(catalog).items():
        base = catalog.get(slug.split("@")[0])
        assert variant.stock_slug == base.slug and variant.inventory_key == base.slug
        assert (variant.length_mm, variant.radius_mm, variant.width_mm, variant.turn_angle_deg) == (
            base.length_mm,
            base.radius_mm,
            base.width_mm,
            base.turn_angle_deg,
        )
        a, b = local_geometry(base, False, TOL), local_geometry(variant, False, TOL)
        assert (a.length_mm, a.turn_rad) == (b.length_mm, b.turn_rad)
        assert a.outline.equals(b.outline)
        assert variant.source == base.source and variant.verified


def test_unverified_pieces_get_no_variant():
    assert variants_of(cached_catalog().get("control_unit")) == []
    assert variants_of(cached_catalog().get("bridge_deck")) == []


def test_flat_problems_are_unchanged_and_bridges_add_variants():
    inventory = {"curve_r1_60": 6, "straight_345": 4}
    flat = Problem(cached_catalog(), inventory, Surface(1500, 800))
    bridged = Problem(cached_catalog(), inventory, Surface(1500, 800), bridges=True)
    assert [p.slug for p in flat.usable] == ["curve_r1_60", "straight_345"]
    assert len(bridged.usable) == 2 + 7 + 7
    assert flat.stock == bridged.stock == inventory
    assert bridged.piece("straight_345@u1").level_out == 2
    with pytest.raises(KeyError):
        flat.piece("straight_345@u1")


# --- inventory: a ramp is a straight, supports may be capped ----------------------------------


def test_a_ramp_element_consumes_a_piece_of_the_stock():
    variants = bridge_variants(cached_catalog())
    validator = IncrementalValidator({"straight_345": 1}, tol=TOL)
    assert not validator.add(variants["straight_345@u0"])
    assert validator.used == {"straight_345": 1, ELEVATED_KEY: 1}
    issues = validator.check_add(variants["straight_345@u1"])
    assert IssueCode.INVENTORY_EXCEEDED in {i.code for i in issues}


def test_the_elevated_cap_is_optional_and_exact():
    catalog = cached_catalog()
    variants = bridge_variants(catalog)
    stock = {"straight_345": 6, "curve_r1_60": 6}
    unlimited = IncrementalValidator(stock, tol=TOL)
    for step in ("@u0", "@u1", "@u2"):
        assert not unlimited.add(variants["straight_345" + step])
    capped = IncrementalValidator({**stock, ELEVATED_KEY: 2}, tol=TOL)
    assert not capped.add(variants["straight_345@u0"]) and not capped.add(
        variants["straight_345@u1"]
    )
    issues = capped.check_add(variants["straight_345@u2"])
    assert {i.code for i in issues} == {IssueCode.INVENTORY_EXCEEDED}
    none_allowed = IncrementalValidator({**stock, ELEVATED_KEY: 0}, tol=TOL)
    assert {i.code for i in none_allowed.check_add(variants["straight_345@u0"])} == {
        IssueCode.INVENTORY_EXCEEDED
    }
    assert not none_allowed.add(catalog.get("straight_345"))  # ground pieces ignore the cap


def test_the_cap_entry_is_not_reported_as_an_ignored_piece():
    inventory = {"curve_r1_60": 6, "straight_345": 4, ELEVATED_KEY: 3}
    assert Problem(cached_catalog(), inventory, Surface(1500, 800), bridges=True).ignored == {}


# --- the staircase: three to climb, three to come down ----------------------------------------


def test_the_climb_takes_three_elements_in_order():
    variants = bridge_variants(cached_catalog())
    validator = IncrementalValidator({"straight_345": 12}, tol=TOL)
    for step in ("@u0", "@u1", "@u2"):
        assert not validator.add(variants["straight_345" + step]), step
    assert validator.end_pose.level == DECK_LEVEL
    assert not validator.add(variants["straight_345@deck"])


@pytest.mark.parametrize("step", ["@u1", "@u2", "@deck", "@d2", "@d1", "@d0"])
def test_from_the_ground_only_the_first_ramp_element_can_start_a_climb(step):
    validator = IncrementalValidator({"straight_345": 2}, tol=TOL)
    issues = validator.check_add(bridge_variants(cached_catalog())["straight_345" + step])
    assert {i.code for i in issues} == {IssueCode.LEVEL_MISMATCH}


def test_one_element_cannot_reach_the_deck_nor_can_there_be_a_second_floor():
    variants = bridge_variants(cached_catalog())
    validator = IncrementalValidator({"straight_345": 12}, tol=TOL)
    assert not validator.add(variants["straight_345@u0"])
    assert IssueCode.LEVEL_MISMATCH in {
        i.code for i in validator.check_add(variants["straight_345@deck"])
    }
    for step in ("@u1", "@u2"):
        assert not validator.add(variants["straight_345" + step])
    assert all(v.level_out <= DECK_LEVEL for v in variants.values())  # no piece goes higher


def test_coming_down_also_takes_three_elements():
    variants = bridge_variants(cached_catalog())
    validator = IncrementalValidator({"straight_345": 12}, tol=TOL)
    for step in ("@u0", "@u1", "@u2", "@deck", "@d2", "@d1"):
        assert not validator.add(variants["straight_345" + step]), step
    assert validator.end_pose.level == 1  # still raised: a loop cannot close here
    assert not validator.add(variants["straight_345@d0"])
    assert validator.end_pose.level == 0


# --- a real bridge: golden_b's crossing, bridged ----------------------------------------------


def test_golden_bridge_loop_is_valid_closed_and_back_at_ground(bridge_problem):
    items = golden_items(bridge_problem)
    result = check(bridge_problem, items)
    assert result.valid, result.issues
    assert result.track.exit_pose.level == 0 and result.track.is_closed(TOL)
    assert result.score_mm == pytest.approx(fixture()["length_mm"])
    elevated = [p for p, _ in items if p.level_in or p.level_out]
    assert [p.slug.partition("@")[2] for p in elevated] == [
        "u0",
        "u1",
        "u2",
        "deck",
        "d2",
        "d1",
        "d0",
    ]
    assert [p.category for p in elevated] == ["straight"] * 2 + ["curve"] + ["straight"] + [
        "curve"
    ] * 2 + ["straight"]


def test_the_same_loop_laid_flat_collides(bridge_problem):
    flat = [(cached_catalog().get(p.inventory_key), m) for p, m in golden_items(bridge_problem)]
    result = validate_layout(
        flat, bridge_problem.inventory, start=golden_start(), tol=TOL, fail_fast=False
    )
    assert IssueCode.COLLISION in result.codes and not result.valid
    assert result.track is not None and result.track.is_closed(
        TOL
    )  # closed, but it overlaps itself


def test_a_two_element_ramp_is_refused(bridge_problem):
    """Dropping one climbing element leaves the deck unreachable."""
    items = golden_items(bridge_problem, replace={11: "straight_345"})
    assert IssueCode.LEVEL_MISMATCH in check(bridge_problem, items).codes


def test_a_loop_cannot_end_raised(bridge_problem):
    items = golden_items(bridge_problem)
    cut = items[:-9]  # stops inside the descent
    result = check(bridge_problem, cut)
    assert not result.valid and result.score_mm == 0.0


def test_the_crossing_must_be_under_the_deck_not_under_a_ramp(bridge_problem):
    """Shifting the bridge one element along puts a ramp element over the crossing: collision."""
    for shift in (-1, 1):
        items = golden_items(bridge_problem, shift=shift)
        result = check(bridge_problem, items, fail_fast=False)
        assert not result.valid and IssueCode.COLLISION in result.codes, shift


def test_the_deck_alone_clears_the_ground(bridge_problem):
    from slotor.geometry import place_piece
    from slotor.validation.rules import bodies_collide, make_body

    variants = bridge_variants(cached_catalog())
    base = cached_catalog().get("straight_345")
    ground = place_piece(Pose(0.0, 0.0, 0.0, 0), base, False, TOL, 0)
    crossing = Pose(150.0, -150.0, math.pi / 2)
    for slug, level, collides in (
        ("straight_345@deck", 3, False),
        ("straight_345@u2", 2, True),
        ("straight_345@u1", 1, True),
        ("straight_345@d2", 3, True),
    ):
        placed = place_piece(
            Pose(crossing.x, crossing.y, crossing.theta, level), variants[slug], False, TOL, 1
        )
        assert bodies_collide(make_body(ground, TOL), make_body(placed, TOL), TOL) is collides, slug


def test_two_decks_cannot_cross_each_other(bridge_problem):
    from slotor.geometry import place_piece
    from slotor.validation.rules import bodies_collide, make_body

    deck = bridge_problem.piece("straight_345@deck")
    a = place_piece(Pose(0.0, 0.0, 0.0, 3), deck, False, TOL, 0)
    b = place_piece(Pose(150.0, -150.0, math.pi / 2, 3), deck, False, TOL, 1)
    assert bodies_collide(make_body(a, TOL), make_body(b, TOL), TOL)


# --- optimizers and exports -------------------------------------------------------------------


def test_beam_and_lns_with_bridges_return_valid_loops(bridge_problem):
    inventory = {k: v for k, v in bridge_problem.inventory.items() if not k.startswith("@")}
    problem = dataclasses.replace(bridge_problem, inventory=inventory)
    flat = beam_search(
        dataclasses.replace(problem, bridges=False),
        seed=0,
        config=BeamConfig(width=60, max_expansions=10_000),
    ).best
    result = lns(problem, seed=0, config=LNSConfig(max_iterations=30), initial=flat)
    assert result.best is not None and flat is not None
    assert_valid_closed(problem, result.best)
    assert result.best.length_mm >= flat.length_mm - 1e-9


def test_lns_starting_from_the_bridge_keeps_a_valid_ground_started_loop(bridge_problem):
    start = make_solution(bridge_problem, golden_items(bridge_problem), golden_start())
    assert start is not None
    result = lns(bridge_problem, seed=2, config=LNSConfig(max_iterations=25), initial=start)
    assert result.best is not None
    assert_valid_closed(bridge_problem, result.best)
    assert result.best.length_mm >= start.length_mm - 1e-9
    assert result.best.start.level == 0


def test_lns_without_bridges_ignores_the_variants(bridge_problem):
    flat_problem = dataclasses.replace(bridge_problem, bridges=False)
    seed = generate_seeds(flat_problem)[0]
    result = lns(flat_problem, seed=1, config=LNSConfig(max_iterations=15), initial=seed)
    assert all(p.level_in == p.level_out == 0 for p, _ in result.best.items)


def test_seeds_stay_flat_with_bridges_allowed(bridge_problem):
    for seed in generate_seeds(bridge_problem)[:3]:
        assert all(p.level_in == p.level_out == 0 for p, _ in seed.items)


def test_report_counts_bridges_ramp_elements_and_decks(bridge_problem):
    solution = make_solution(bridge_problem, golden_items(bridge_problem), golden_start())
    report = build_report(solution, bridge_problem)
    assert report.elevated == {"bridges": 1, "ramp": 6, "raised": 1}
    summary = format_summary(report)
    assert "1 pont(s)" in summary and "Supports" in summary
    bom = bill_of_materials(report)
    assert "(ramp up 1/3)" in bom and "(raised)" in bom and "(ramp down 3/3)" in bom
    assert f"Total : {solution.n_pieces}" in bom
    assert all(row.used <= row.available for row in report.usage)  # variants count on their base


def test_json_round_trip_keeps_the_variants_and_revalidates(bridge_problem):
    solution = make_solution(bridge_problem, golden_items(bridge_problem), golden_start())
    text = solution_to_json(solution, bridge_problem)
    data = json.loads(text)
    assert data["bridges"] is True and any("@u0" in p["slug"] for p in data["pieces"])
    assert solution_from_json(text, bridge_problem).slugs == solution.slugs
    flat_problem = dataclasses.replace(bridge_problem, bridges=False)
    with pytest.raises(ValueError):  # a bridge file cannot be loaded as a flat problem
        solution_from_json(text, flat_problem)


def test_svg_draws_ramps_and_decks_over_the_ground(bridge_problem):
    solution = make_solution(bridge_problem, golden_items(bridge_problem), golden_start())
    svg = solution_to_svg(solution, bridge_problem)
    assert svg.count("<polygon") == solution.n_pieces
    assert "#c9a15a" in svg and "#a07a3a" in svg and "#ffd166" in svg
    assert svg.find('fill="#a07a3a"') > svg.rfind('fill="#6f7c99"')  # ground first, bridge over it


def test_service_builds_bridge_problems_with_an_optional_cap():
    base = dict(
        surface_width_mm=2700.0,
        surface_height_mm=1700.0,
        margin_mm=0.0,
        inventory={"curve_r1_60": 12, "straight_345": 21},
    )
    problem = build_problem(cached_catalog(), OptimizeRequest(**base, allow_bridges=True))
    assert problem.bridges and ELEVATED_KEY not in problem.inventory
    capped = build_problem(
        cached_catalog(), OptimizeRequest(**base, allow_bridges=True, max_elevated=7)
    )
    assert capped.inventory[ELEVATED_KEY] == 7
    zero = build_problem(
        cached_catalog(), OptimizeRequest(**base, allow_bridges=True, max_elevated=0)
    )
    assert zero.inventory[ELEVATED_KEY] == 0
    with pytest.raises(ValueError, match="max_elevated"):
        build_problem(cached_catalog(), OptimizeRequest(**base, max_elevated=-1))


# --- property: incremental == full, with levels in play ---------------------------------------

_ALL = {
    **{p.slug: p for p in cached_catalog().pieces if p.slug in ("straight_345", "curve_r1_60")},
    **{
        slug: v
        for slug, v in bridge_variants(cached_catalog()).items()
        if slug.split("@")[0] in ("straight_345", "curve_r1_60")
    },
}
_SLUGS = sorted(_ALL)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(st.sampled_from(_SLUGS), st.booleans(), st.integers(1, 4)),
        min_size=1,
        max_size=8,
    ),
    st.one_of(st.none(), st.integers(0, 8)),
)
def test_incremental_agrees_with_full_when_levels_are_involved(runs, cap):
    inventory = {"straight_345": 30, "curve_r1_60": 30}
    if cap is not None:
        inventory[ELEVATED_KEY] = cap
    chain = [(_ALL[slug], flip and "curve" in slug) for slug, flip, n in runs for _ in range(n)]
    inc = IncrementalValidator(inventory, tol=TOL)
    accepted = []
    for piece, mirrored in chain:
        verdict = inc.check_add(piece, mirrored)
        full = validate_layout(
            [*accepted, (piece, mirrored)], inventory, require_closed=False, tol=TOL
        )
        assert (not verdict) == full.valid, (verdict, full.issues)
        assert {i.code for i in verdict} == set(full.codes)
        if not verdict:
            assert not inc.add(piece, mirrored)
            accepted.append((piece, mirrored))


# --- crossing -> bridge conversion (D31) ---------------------------------------------------------


def golden_b_flat_items():
    """The flat golden_b chain (crossing included) built from the bridge fixture's layout."""
    catalog = cached_catalog()
    return [(catalog.get(slug.split("@")[0]), mirrored) for slug, mirrored in fixture()["sequence"]]


def test_the_flat_golden_b_has_exactly_one_crossing_pair():
    from slotor.optimizers.bridging import overlapping_pairs

    pairs = overlapping_pairs(golden_b_flat_items(), TOL)
    assert len(pairs) == 1
    deck_index = fixture()["bridge_indices"][3]
    assert deck_index in pairs[0]


def test_conversion_turns_the_crossing_into_a_valid_bridge(bridge_problem):
    from slotor.optimizers import convert_to_bridges

    results = convert_to_bridges(bridge_problem, golden_b_flat_items())
    assert results, "the crossing of golden_b must be convertible"
    best = results[0]
    assert_valid_closed(bridge_problem, best)
    assert best.length_mm == pytest.approx(fixture()["length_mm"])
    elevated = [p.slug.partition("@")[2] for p, _ in best.items if p.level_in or p.level_out]
    assert sorted(elevated) == sorted(["u0", "u1", "u2", "deck", "d2", "d1", "d0"])
    assert best.start.level == 0


def test_conversion_leaves_a_crossing_free_loop_flat():
    from slotor.optimizers import convert_to_bridges

    problem = Problem(
        cached_catalog(), {"curve_r1_60": 6, "straight_345": 4}, Surface(1500, 800), bridges=True
    )
    flat = generate_seeds(problem)[0]
    results = convert_to_bridges(problem, list(flat.items))
    assert results and all(p.level_in == p.level_out == 0 for p, _ in results[0].items)


def test_a_window_with_a_short_piece_cannot_be_lifted(bridge_problem):
    from slotor.optimizers.bridging import _lift

    items = golden_b_flat_items()
    deck = (fixture()["bridge_indices"][3],)
    assert _lift(bridge_problem, items, deck) is not None
    items[deck[0] - 2] = (cached_catalog().get("straight_115"), False)
    assert _lift(bridge_problem, items, deck) is None  # a 1/3 straight cannot climb


def test_a_loop_too_short_for_a_bridge_cannot_be_lifted(bridge_problem):
    from slotor.optimizers.bridging import _lift

    straight = cached_catalog().get("straight_345")
    assert _lift(bridge_problem, [(straight, False)] * 6, (0,)) is None


def test_bridge_seeds_exist_only_when_bridges_are_allowed(bridge_problem):
    from slotor.optimizers import bridge_seeds

    flat = dataclasses.replace(bridge_problem, bridges=False)
    assert bridge_seeds(flat) == []
    seeds = bridge_seeds(bridge_problem)
    assert seeds and all(s.validation.valid for s in seeds)
    assert seeds[0].length_mm == pytest.approx(fixture()["length_mm"])


def test_a_bridged_seed_beats_every_flat_seed_on_the_golden_b_inventory():
    from slotor.optimizers import bridge_seeds

    inventory = {"curve_r1_60": 12, "straight_345": 16}
    problem = Problem(cached_catalog(), inventory, Surface(2680, 1840), bridges=True)
    flat_best = generate_seeds(problem)[0].length_mm
    seeds = bridge_seeds(problem)
    assert seeds and seeds[0].length_mm > flat_best + 1000.0
    for seed in seeds:
        assert_valid_closed(problem, seed)
        assert sum(1 for p, _ in seed.items if p.level_in or p.level_out) == 7


def test_the_elevated_cap_limits_the_bridged_seeds():
    from slotor.optimizers import bridge_seeds

    inventory = {"curve_r1_60": 12, "straight_345": 16, ELEVATED_KEY: 6}
    problem = Problem(cached_catalog(), inventory, Surface(2680, 1840), bridges=True)
    assert bridge_seeds(problem) == []  # a bridge needs seven elevated elements


def test_lns_starts_from_the_bridged_seed_and_never_loses_it():
    inventory = {"curve_r1_60": 12, "straight_345": 16}
    problem = Problem(cached_catalog(), inventory, Surface(2680, 1840), bridges=True)
    result = lns(problem, seed=0, config=LNSConfig(max_iterations=20))
    assert result.best is not None
    assert_valid_closed(problem, result.best)
    assert result.best.length_mm >= 8900.0
    assert any(p.level_in or p.level_out for p, _ in result.best.items)


def test_the_pipeline_produces_a_bridge_and_beats_the_flat_pipeline():
    from slotor.service import optimize

    inventory = {"curve_r1_60": 12, "straight_345": 16}
    flat = Problem(cached_catalog(), inventory, Surface(2680, 1840))
    bridged = dataclasses.replace(flat, bridges=True)
    flat_best = optimize(flat, seed=0, time_budget_s=4.0, top_k=2).best
    result = optimize(bridged, seed=0, time_budget_s=4.0, top_k=2)
    assert result.best is not None and flat_best is not None
    assert_valid_closed(bridged, result.best)
    assert result.best.length_mm > flat_best.length_mm + 300.0
    assert any(stage == "bridge" for stage, _ in result.stages)
