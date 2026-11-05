"""Catalog dimensions must reproduce the bounding boxes Track Architect displays (D22).

These are the tool's own geometry, not physical measurements. The display resolution is 0.1 cm, so a
predicted value may differ from the displayed one by half a unit (plus float slack).
"""

from __future__ import annotations

import json
import math

import pytest

from slotor.catalog import Catalog
from slotor.catalog.seed import read_seed

from .conftest import FIXTURES

_DISPLAY_TOL_CM = 0.06  # 0.05 cm rounding of a 0.1 cm display + float slack


def _fixture() -> dict:
    return json.loads((FIXTURES / "ta_piece_bboxes.json").read_text(encoding="utf-8"))


def _identified() -> list[dict]:
    return [p for p in _fixture()["pieces"] if p["slug"] and p["slug"] != "control_unit"]


def _predicted_bbox_cm(catalog: Catalog, slug: str, orientation: str) -> tuple[float, float]:
    piece = catalog.get(slug)
    if piece.category == "straight":
        return piece.length_mm / 10, piece.width_mm / 10
    assert piece.radius_ref == "outer_edge"
    outer = piece.radius_mm
    inner = outer - piece.width_mm
    theta = math.radians(piece.turn_angle_deg)
    if orientation == "symmetric":  # sector centred on the turn axis, centre below the piece
        width = 2 * outer * math.sin(theta / 2)
        height = outer - inner * math.cos(theta / 2)
    elif orientation == "start_face_vertical":  # entry face on the vertical radius
        width = outer * math.sin(theta)
        height = outer - inner * math.cos(theta)
    else:  # pragma: no cover - guards the fixture vocabulary
        raise AssertionError(orientation)
    return width / 10, height / 10


@pytest.mark.parametrize("entry", _identified(), ids=lambda e: e["label"])
def test_catalog_reproduces_displayed_bounding_box(catalog, entry):
    width, height = _predicted_bbox_cm(catalog, entry["slug"], entry["orientation"])
    shown_w, shown_h = entry["bbox_cm"]
    assert width == pytest.approx(shown_w, abs=_DISPLAY_TOL_CM)
    assert height == pytest.approx(shown_h, abs=_DISPLAY_TOL_CM)


def test_inner_edge_reading_is_rejected_by_the_data(catalog):
    """Regression guard for D22: reading the radii as inner edges misses every displayed box."""
    for entry in (e for e in _identified() if e["slug"].startswith("curve_")):
        piece = catalog.get(entry["slug"])
        outer = piece.radius_mm + piece.width_mm  # what an inner-edge reading would imply
        inner = piece.radius_mm
        theta = math.radians(piece.turn_angle_deg)
        if entry["orientation"] == "symmetric":
            wrong_width = 2 * outer * math.sin(theta / 2) / 10
        else:
            wrong_width = outer * math.sin(theta) / 10
        assert abs(wrong_width - entry["bbox_cm"][0]) > 1.0, (entry["label"], inner)


def test_control_unit_extends_beyond_the_rail():
    cu = next(p for p in _fixture()["pieces"] if p["slug"] == "control_unit")
    assert cu["bbox_cm"][0] == 34.5  # along the track, one standard straight
    assert cu["bbox_cm"][1] > 19.8  # wider than the rail: the block sticks out (footprint)


def test_every_unidentified_piece_is_flagged_for_confirmation():
    unidentified = [p for p in _fixture()["pieces"] if p["slug"] is None]
    assert unidentified, "fixture should list the special pieces awaiting identification"
    assert all(p["bbox_cm"][0] > 0 and p["bbox_cm"][1] > 0 for p in unidentified)


def test_user_supplied_pack_refs_exist_in_the_seed():
    seed, _ = read_seed()
    known = {pack["ref"] for pack in seed["packs"]}
    refs = {p["ref"] for p in _fixture()["pieces"] if p.get("ref")}
    assert refs, "special pieces should carry the pack refs supplied by the user"
    assert refs <= known, sorted(refs - known)
