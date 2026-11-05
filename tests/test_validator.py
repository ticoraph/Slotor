"""Negative and positive tests of the single validator (Phase 2, D6)."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from slotor.geometry import Pose, ToleranceConfig, build_track, place_piece
from slotor.validation import IssueCode, Surface, validate_layout
from slotor.validation.rules import bodies_collide, make_body

from .conftest import FIXTURES

TOL = ToleranceConfig()
OVAL_INVENTORY = {"curve_r1_60": 6, "straight_345": 4}
OVAL_LENGTH_MM = 6 * 297.0 * math.pi / 3 + 4 * 345.0


def oval(catalog, *, flip: bool = False):
    curve, straight = catalog.get("curve_r1_60"), catalog.get("straight_345")
    items = [(curve, flip)] * 3 + [(straight, False)] * 2 + [(curve, flip)] * 3
    return items + [(straight, False)] * 2


def start_inside(items, surface_margin: float = 0.0, extra: float = 0.0) -> Pose:
    """A start pose that puts the layout's bounding box at the lower-left of the surface."""
    minx, miny, _, _ = build_track(items).bounds
    return Pose(surface_margin + extra - minx, surface_margin + extra - miny, 0.0, 0)


def codes(result) -> set[IssueCode]:
    return set(result.codes)


def test_oval_is_valid_and_scores_its_centerline_length(catalog):
    items = oval(catalog)
    result = validate_layout(items, OVAL_INVENTORY, start=start_inside(items))
    assert result.valid, result.issues
    assert result.score_mm == pytest.approx(OVAL_LENGTH_MM)


def test_empty_layout_is_invalid_when_a_loop_is_required():
    assert codes(validate_layout([], {})) == {IssueCode.EMPTY}
    assert validate_layout([], {}, require_closed=False).valid


def test_unverified_piece_is_refused(catalog):
    items = [(catalog.get("control_unit"), False)]
    result = validate_layout(items, {"control_unit": 1})
    assert IssueCode.UNVERIFIED_PIECE in result.codes
    assert result.score_mm == 0.0


def test_inventory_exceeded(catalog):
    result = validate_layout(oval(catalog), {"curve_r1_60": 5, "straight_345": 4})
    assert codes(result) == {IssueCode.INVENTORY_EXCEEDED}
    assert result.score_mm == 0.0


def test_piece_absent_from_the_inventory_is_refused(catalog):
    result = validate_layout(oval(catalog), {"curve_r1_60": 6})
    assert codes(result) == {IssueCode.INVENTORY_EXCEEDED}


def test_unknown_dimension_is_a_geometry_issue(catalog):
    broken = dataclasses.replace(catalog.get("straight_345"), length_mm=None)
    result = validate_layout([(broken, False)], {"straight_345": 1}, require_closed=False)
    assert codes(result) == {IssueCode.GEOMETRY}


def test_level_mismatch(catalog):
    raised = dataclasses.replace(catalog.get("straight_345"), level_in=1, level_out=1)
    result = validate_layout([(raised, False)], {"straight_345": 1}, require_closed=False)
    assert codes(result) == {IssueCode.LEVEL_MISMATCH}


def test_open_loop(catalog):
    items = [(catalog.get("curve_r1_60"), False)] * 5
    result = validate_layout(items, {"curve_r1_60": 5})
    assert codes(result) == {IssueCode.NOT_CLOSED}
    assert result.score_mm == 0.0


def test_partial_chain_is_valid_without_closure(catalog):
    items = [(catalog.get("curve_r1_60"), False)] * 5
    assert validate_layout(items, {"curve_r1_60": 5}, require_closed=False).valid


def test_gap_just_inside_the_closure_tolerance_closes(catalog):
    """A 9 mm gap is within the experienced 10 mm; 11 mm is not."""
    straight, curve = catalog.get("straight_345"), catalog.get("curve_r1_60")
    for shortfall, expected in ((9.0, True), (11.0, False)):
        short = dataclasses.replace(straight, slug="short", length_mm=345.0 - shortfall / 2)
        items = [(curve, False)] * 3 + [(straight, False)] * 2 + [(curve, False)] * 3
        items += [(short, False)] * 2  # only one side is shorter, so the oval is left open
        result = validate_layout(items, {"short": 2, "straight_345": 2, "curve_r1_60": 6})
        assert result.valid is expected, (shortfall, result.issues)


def test_loop_left_at_level_one_is_refused(catalog):
    raised = dataclasses.replace(catalog.get("curve_r1_60"), slug="raised", level_in=1, level_out=1)
    start = Pose(0.0, 0.0, 0.0, 1)
    result = validate_layout([(raised, False)] * 6, {"raised": 6}, start=start)
    assert codes(result) == {IssueCode.NOT_AT_GROUND}


def test_surface_exact_fit_and_one_millimetre_short(catalog):
    items = oval(catalog)
    start = start_inside(items)
    width, height = 1482.0, 792.0
    fits = validate_layout(items, OVAL_INVENTORY, start=start, surface=Surface(width, height))
    assert fits.valid, fits.issues
    too_small = Surface(width - 1.0, height)
    result = validate_layout(items, OVAL_INVENTORY, start=start, surface=too_small)
    assert IssueCode.OUT_OF_SURFACE in result.codes


def test_surface_margin_applies(catalog):
    items = oval(catalog)
    surface = Surface(1482.0, 792.0, margin_mm=20.0)
    result = validate_layout(items, OVAL_INVENTORY, start=start_inside(items), surface=surface)
    assert IssueCode.OUT_OF_SURFACE in result.codes
    centered = start_inside(items, surface_margin=20.0)
    ok = validate_layout(
        items, OVAL_INVENTORY, start=centered, surface=Surface(1522.0, 832.0, 20.0)
    )
    assert ok.valid, ok.issues


def test_circle_revisited_collides(catalog):
    curve = catalog.get("curve_r1_60")
    items = [(curve, False)] * 7
    result = validate_layout(items, {"curve_r1_60": 7}, require_closed=False)
    assert codes(result) == {IssueCode.COLLISION}


def test_mirrored_oval_is_valid_too(catalog):
    items = oval(catalog, flip=True)
    result = validate_layout(items, OVAL_INVENTORY, start=start_inside(items))
    assert result.valid, result.issues


# --- near-contact: the calibrated depth (D24) ---------------------------------------------------


def _two_straights(catalog, gap_mm: float):
    """Two parallel straights; ``gap_mm`` is the distance between their facing edges."""
    straight = catalog.get("straight_345")
    width = straight.width_mm
    first = place_piece(Pose(0.0, 0.0, 0.0, 0), straight, False, TOL, 0)
    second = place_piece(Pose(0.0, width + gap_mm, 0.0, 0), straight, False, TOL, 1)
    return make_body(first, TOL), make_body(second, TOL)


@pytest.mark.parametrize(
    ("gap_mm", "collides"),
    [(5.0, False), (0.0, False), (-1.0, False), (-1.9, False), (-2.5, True), (-10.0, True)],
)
def test_side_by_side_rails_collide_only_beyond_the_penetration_depth(catalog, gap_mm, collides):
    a, b = _two_straights(catalog, gap_mm)
    assert bodies_collide(a, b, TOL) is collides


def test_neighbours_in_a_chain_do_not_collide(catalog):
    items = [(catalog.get("curve_r1_60"), False)] * 6
    assert validate_layout(items, {"curve_r1_60": 6}).valid


# --- two levels (D4), synthetic pieces: the catalog has none yet ---------------------------------


def _level_pieces(catalog):
    base = catalog.get("straight_345")
    deck = dataclasses.replace(base, slug="deck", level_in=3, level_out=3)
    ramp = dataclasses.replace(base, slug="ramp", level_in=0, level_out=1)
    return base, deck, ramp


def _crossing_bodies(catalog, upper, lower, upper_level, lower_level):
    first = place_piece(Pose(0.0, 0.0, 0.0, lower_level), lower, False, TOL, 0)
    second = place_piece(Pose(150.0, -150.0, math.pi / 2, upper_level), upper, False, TOL, 1)
    return make_body(first, TOL), make_body(second, TOL)


def test_deck_over_ground_piece_is_allowed(catalog):
    base, deck, _ = _level_pieces(catalog)
    a, b = _crossing_bodies(catalog, deck, base, 3, 0)
    assert not bodies_collide(a, b, TOL)


def test_same_level_crossing_collides(catalog):
    base, _, _ = _level_pieces(catalog)
    a, b = _crossing_bodies(catalog, base, base, 0, 0)
    assert bodies_collide(a, b, TOL)


def test_two_decks_crossing_collide(catalog):
    _, deck, _ = _level_pieces(catalog)
    a, b = _crossing_bodies(catalog, deck, deck, 3, 3)
    assert bodies_collide(a, b, TOL)


def test_a_ramp_collides_with_anything_it_overlaps(catalog):
    base, deck, ramp = _level_pieces(catalog)
    for other, level in ((base, 0), (deck, 3)):
        a, b = _crossing_bodies(catalog, ramp, other, 0, level)
        assert bodies_collide(a, b, TOL), other.slug
        a, b = _crossing_bodies(catalog, other, ramp, level, 0)
        assert bodies_collide(a, b, TOL), other.slug


# --- golden circuits ----------------------------------------------------------------------------


def _golden(catalog, name):
    data = json.loads((FIXTURES / "golden_circuits.json").read_text(encoding="utf-8"))
    circuit = next(c for c in data["circuits"] if c["name"] == name)
    proxies = data["geometry_proxy"]
    items = []
    for slug, flip in circuit["sequence"]:
        base = catalog.get(proxies.get(slug, slug))
        piece = dataclasses.replace(base, slug=slug) if slug in proxies else base
        items.append((piece, flip == "R"))
    inventory = {}
    for piece, _ in items:
        inventory[piece.slug] = inventory.get(piece.slug, 0) + 1
    st = circuit["start"]
    start = Pose(st["x"], st["y"], math.radians(st["heading_deg"]), 0)
    minx, miny, maxx, maxy = build_track(items, start=start).bounds
    shifted = Pose(start.x - minx, start.y - miny, start.theta, 0)
    return circuit, items, inventory, shifted, (maxx - minx, maxy - miny)


def test_golden_a_is_valid(catalog):
    circuit, items, inventory, start, (width, height) = _golden(catalog, "golden_a")
    surface = Surface(width, height + circuit["control_unit_overhang_mm"])
    result = validate_layout(items, inventory, start=start, surface=surface)
    assert result.valid, result.issues
    assert result.score_mm > 0.0


def test_golden_b_is_rejected_for_its_2d_crossing(catalog):
    _, items, inventory, start, _ = _golden(catalog, "golden_b")
    result = validate_layout(items, inventory, start=start, fail_fast=False)
    assert IssueCode.COLLISION in result.codes
    assert result.score_mm == 0.0
    assert result.track is not None and result.track.is_closed(TOL)  # closed, but it overlaps


def test_golden_a_with_one_piece_missing_is_not_closed(catalog):
    _, items, inventory, start, _ = _golden(catalog, "golden_a")
    result = validate_layout(items[:-1], inventory, start=start)
    assert IssueCode.NOT_CLOSED in result.codes
