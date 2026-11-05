"""Golden circuits rebuilt by hand from Track Architect captures (D13, Phase 1 exit criteria)."""

from __future__ import annotations

import dataclasses
import json
import math
from collections import Counter

import pytest

from slotor.geometry import Pose, ToleranceConfig, build_track, track_to_svg

from .conftest import FIXTURES


def _data() -> dict:
    return json.loads((FIXTURES / "golden_circuits.json").read_text(encoding="utf-8"))


def _build(catalog, circuit: dict, proxies: dict[str, str]):
    items = []
    for slug, flip in circuit["sequence"]:
        base = catalog.get(proxies.get(slug, slug))
        piece = dataclasses.replace(base, slug=slug) if slug in proxies else base
        items.append((piece, flip == "R"))
    start = circuit["start"]
    pose = Pose(start["x"], start["y"], math.radians(start["heading_deg"]), 0)
    return build_track(items, start=pose, tol=ToleranceConfig())


@pytest.fixture(params=_data()["circuits"], ids=lambda c: c["name"])
def circuit(request):
    return request.param


def test_sequence_matches_the_ta_inventory(circuit, golden_inventories):
    inventory = next(i for i in golden_inventories if i["name"] == circuit["inventory"])
    counts = Counter(slug for slug, _ in circuit["sequence"])
    assert dict(counts) == inventory["counts"]


def test_golden_circuit_closes(catalog, circuit):
    track = _build(catalog, circuit, _data()["geometry_proxy"])
    assert track.is_closed(ToleranceConfig()), track.closure_error()


def test_total_turn_matches_the_layout(catalog, circuit):
    track = _build(catalog, circuit, _data()["geometry_proxy"])
    assert track.total_turn_deg == pytest.approx(circuit["expected_total_turn_deg"], abs=1e-9)


def test_footprint_is_close_to_track_architect(catalog, circuit):
    """golden_a matches the displayed box within its rounding; golden_b is ~0.5 % off (see D23)."""
    track = _build(catalog, circuit, _data()["geometry_proxy"])
    minx, miny, maxx, maxy = track.bounds
    width, height = maxx - minx, maxy - miny + circuit["control_unit_overhang_mm"]
    shown_w, shown_h = circuit["ta_bbox_mm"]
    tolerance = circuit["bbox_tolerance_mm"]
    assert width == pytest.approx(shown_w, abs=tolerance)
    assert height == pytest.approx(shown_h, abs=tolerance)


def test_golden_a_residual_gap_is_the_nominal_one_third_straight_mismatch(catalog):
    """The 1.19 mm gap is exactly 2 x 115 x sin(60 deg) - 198: nominal geometry, not a bug."""
    circuit = next(c for c in _data()["circuits"] if c["name"] == "golden_a")
    track = _build(catalog, circuit, _data()["geometry_proxy"])
    error = track.closure_error()
    assert error.dx == pytest.approx(0.0, abs=1e-9)
    expected = 2 * 115.0 * math.sin(math.pi / 3) - 198.0
    assert abs(error.dy) == pytest.approx(expected, abs=1e-9)
    assert not track.is_closed(
        ToleranceConfig(closure_pos_mm=1e-3)
    )  # a numeric-only check refuses it


def test_svg_debug_rendering_has_one_polygon_per_piece(catalog, circuit):
    track = _build(catalog, circuit, _data()["geometry_proxy"])
    svg = track_to_svg(track)
    assert svg.count("<polygon") == len(circuit["sequence"])
