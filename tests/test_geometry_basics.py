"""Hand-computed checks of poses, straights, curves, mirroring and closure (Phase 1)."""

from __future__ import annotations

import dataclasses
import functools
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from slotor.catalog import import_seed, load_catalog
from slotor.db import apply_migrations, connect
from slotor.geometry import (
    GeometryError,
    LevelMismatchError,
    Pose,
    ToleranceConfig,
    build_track,
    local_geometry,
)

TOL = ToleranceConfig()


def test_pose_rotates_local_offsets():
    pose = Pose(10.0, 20.0, math.pi / 2, 0)
    moved = pose.advance(100.0, 0.0, 0.0)
    assert moved.x == pytest.approx(10.0, abs=1e-9)
    assert moved.y == pytest.approx(120.0)
    assert moved.theta == pytest.approx(math.pi / 2)


def test_straight_exit_and_length(catalog):
    geom = local_geometry(catalog.get("straight_345"), mirrored=False, tol=TOL)
    assert geom.length_mm == 345.0
    assert geom.exit == Pose(345.0, 0.0, 0.0, 0)
    assert geom.outline.area == pytest.approx(345.0 * 198.0)


def test_left_curve_exit_matches_hand_calculation(catalog):
    # R1 60 deg, outer-edge radius 396 -> centerline radius 297 (D22).
    geom = local_geometry(catalog.get("curve_r1_60"), mirrored=False, tol=TOL)
    rc = 297.0
    assert geom.exit.x == pytest.approx(rc * math.sin(math.pi / 3))
    assert geom.exit.y == pytest.approx(rc * (1 - math.cos(math.pi / 3)))
    assert geom.exit.theta == pytest.approx(math.pi / 3)
    assert geom.length_mm == pytest.approx(rc * math.pi / 3)


def test_mirrored_curve_is_the_reflection(catalog):
    piece = catalog.get("curve_r2_30")
    left = local_geometry(piece, mirrored=False, tol=TOL)
    right = local_geometry(piece, mirrored=True, tol=TOL)
    assert right.exit.x == pytest.approx(left.exit.x)
    assert right.exit.y == pytest.approx(-left.exit.y)
    assert right.exit.theta == pytest.approx(-left.exit.theta)
    assert right.length_mm == pytest.approx(left.length_mm)
    assert right.outline.area == pytest.approx(left.outline.area)


def test_curve_outline_spans_the_rail_width(catalog):
    geom = local_geometry(catalog.get("curve_r1_60"), mirrored=False, tol=TOL)
    # Annular sector between radii 198 and 396 over 60 deg (inscribed polygon: within sagitta).
    exact = 0.5 * (396.0**2 - 198.0**2) * math.pi / 3
    assert geom.outline.area == pytest.approx(exact, rel=1e-3)


def test_straight_is_unaffected_by_mirroring(catalog):
    piece = catalog.get("straight_115")
    assert local_geometry(piece, mirrored=True, tol=TOL).exit == Pose(115.0, 0.0, 0.0, 0)


def test_missing_dimension_is_refused(catalog):
    piece = dataclasses.replace(catalog.get("straight_345"), length_mm=None)
    with pytest.raises(GeometryError, match="length_mm"):
        local_geometry(piece, mirrored=False, tol=TOL)


def test_unknown_radius_reference_is_refused(catalog):
    piece = dataclasses.replace(catalog.get("curve_r1_60"), radius_ref="lane1")
    with pytest.raises(GeometryError, match="radius_ref"):
        local_geometry(piece, mirrored=False, tol=TOL)


def test_unsupported_category_is_refused(catalog):
    with pytest.raises(GeometryError, match="control_unit"):
        local_geometry(catalog.get("control_unit"), mirrored=False, tol=TOL)


def test_level_mismatch_is_reported(catalog):
    raised = dataclasses.replace(catalog.get("straight_345"), level_in=1, level_out=1)
    with pytest.raises(LevelMismatchError):
        build_track([(raised, False)])


def test_six_60_degree_curves_close_a_circle(catalog):
    curve = catalog.get("curve_r1_60")
    track = build_track([(curve, False)] * 6)
    assert track.is_closed(TOL)
    assert track.total_turn_deg == pytest.approx(360.0)
    assert track.closure_error().distance_mm < 1e-6


def test_default_closure_tolerance_is_the_experienced_10_mm():
    assert TOL.closure_pos_mm == 10.0


def test_closure_angle_tolerance_is_one_millimetre_at_the_rail_edge(catalog):
    curve = catalog.get("curve_r1_60")
    full = [(curve, False)] * 6
    allowed = TOL.closure_angle_edge_mm / 198.0
    start = Pose(0.0, 0.0, 0.0, 0)

    def closes_with_heading_error(error: float) -> bool:
        track = build_track(full, start=Pose(start.x, start.y, error, 0))
        # Shift the start heading so the exit differs from it only by the heading error.
        return dataclasses.replace(track, start=start).is_closed(TOL)

    assert closes_with_heading_error(0.9 * allowed)
    assert not closes_with_heading_error(1.1 * allowed)


def test_five_curves_do_not_close(catalog):
    track = build_track([(catalog.get("curve_r1_60"), False)] * 5)
    assert not track.is_closed(TOL)


def test_oval_closes_and_has_expected_bounding_box(catalog):
    """Control oval E2 of the measurement sheet: 6 x R1 60 + 4 x straight 34.5."""
    curve, straight = catalog.get("curve_r1_60"), catalog.get("straight_345")
    items = [(curve, False)] * 3 + [(straight, False)] * 2 + [(curve, False)] * 3
    items += [(straight, False)] * 2
    track = build_track(items)
    assert track.is_closed(TOL)
    minx, miny, maxx, maxy = track.bounds
    assert maxx - minx == pytest.approx(1482.0, abs=0.1)
    assert maxy - miny == pytest.approx(792.0, abs=0.1)


def test_lane_lengths_follow_the_arc_radius(catalog):
    curve = catalog.get("curve_r1_60")
    track = build_track([(curve, False)] * 6)
    offset = 30.0  # arbitrary test offset: the real lane spacing is not in the catalog
    inner = track.lane_length_mm(+offset)  # left-hand turns: +offset is the inner lane
    outer = track.lane_length_mm(-offset)
    assert inner == pytest.approx((297.0 - offset) * 2 * math.pi)
    assert outer == pytest.approx((297.0 + offset) * 2 * math.pi)
    assert track.length_mm == pytest.approx(297.0 * 2 * math.pi)


def test_lane_length_refuses_an_offset_beyond_the_radius(catalog):
    track = build_track([(catalog.get("curve_r1_60"), False)] * 6)
    with pytest.raises(GeometryError):
        track.lane_length_mm(300.0)


_SLUGS = ["curve_r1_60", "curve_r2_30", "straight_345", "straight_115"]


@given(st.lists(st.tuples(st.sampled_from(_SLUGS), st.booleans()), max_size=12))
def test_total_turn_is_the_sum_of_signed_angles(raw):
    catalog = _catalog_once()
    items = [(catalog.get(slug), flip) for slug, flip in raw]
    track = build_track(items)
    expected = sum(
        (p.turn_angle_deg or 0.0) * (-1 if flip and p.category == "curve" else 1)
        for p, flip in items
    )
    assert track.total_turn_deg == pytest.approx(expected)
    heading_gap = track.exit_pose.theta - math.radians(expected)
    assert abs(math.remainder(heading_gap, 2 * math.pi)) < 1e-9


@given(st.lists(st.sampled_from(_SLUGS), min_size=1, max_size=10), st.floats(0.0, 6.28))
def test_rotating_the_start_rotates_the_whole_track(slugs, angle):
    """Invariance under rigid motion: lengths and turns do not depend on the start pose."""
    catalog = _catalog_once()
    items = [(catalog.get(s), False) for s in slugs]
    base = build_track(items)
    moved = build_track(items, start=Pose(123.0, -45.0, angle, 0))
    assert moved.length_mm == pytest.approx(base.length_mm)
    assert moved.total_turn_deg == pytest.approx(base.total_turn_deg)
    assert moved.closure_error().distance_mm == pytest.approx(base.closure_error().distance_mm)


@functools.cache
def _catalog_once():
    conn = connect(":memory:")
    apply_migrations(conn)
    import_seed(conn)
    name = conn.execute("SELECT name FROM catalog_version").fetchone()["name"]
    return load_catalog(conn, name)
