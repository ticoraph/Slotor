"""Local geometry of a catalog piece: connectors, centerline length and body outline.

Frame convention: the entry connector is the origin, heading ``+x``. A left-hand curve turns towards
``+y`` (counter-clockwise). The outline is the body of the rail only; borders are excluded (D20).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from shapely.geometry import Polygon

from slotor.catalog.model import PieceType
from slotor.geometry.pose import Pose
from slotor.geometry.tolerance import ToleranceConfig


class GeometryError(ValueError):
    """The piece cannot be placed: unknown dimension, unsupported kind or inconsistent data."""


class LevelMismatchError(GeometryError):
    """A piece's entry level differs from the level of the track end it is attached to."""


@dataclass(frozen=True, slots=True)
class LocalGeometry:
    slug: str
    kind: str  # "straight" or "curve"
    entry: Pose
    exit: Pose
    length_mm: float  # centerline length
    turn_rad: float  # signed, positive = left
    centerline_radius_mm: float | None  # None for a straight
    width_mm: float
    outline: Polygon


def _require(piece: PieceType, *names: str) -> None:
    missing = [n for n in names if getattr(piece, n) is None]
    if missing:
        raise GeometryError(
            f"piece {piece.slug!r} lacks {', '.join(missing)}: refusing to guess (D5)"
        )


def centerline_radius_mm(piece: PieceType) -> float:
    """Centerline radius from the catalog radius and its reference line (D20, D22)."""
    _require(piece, "radius_mm", "width_mm", "radius_ref")
    half = piece.width_mm / 2  # type: ignore[operator]
    if piece.radius_ref == "outer_edge":
        return piece.radius_mm - half  # type: ignore[operator]
    if piece.radius_ref == "inner_edge":
        return piece.radius_mm + half  # type: ignore[operator]
    if piece.radius_ref == "centerline":
        return piece.radius_mm  # type: ignore[return-value]
    raise GeometryError(f"piece {piece.slug!r}: unsupported radius_ref {piece.radius_ref!r}")


def _arc_points(
    radius: float, start_angle: float, sweep: float, sagitta: float
) -> list[tuple[float, float]]:
    """Arc points about the origin, chord sagitta <= ``sagitta``, both ends included."""
    ratio = min(1.0, sagitta / radius)
    max_step = 2 * math.acos(1 - ratio) if ratio < 1 else math.pi
    n = max(1, math.ceil(abs(sweep) / max_step))
    return [
        (
            radius * math.cos(start_angle + sweep * i / n),
            radius * math.sin(start_angle + sweep * i / n),
        )
        for i in range(n + 1)
    ]


@lru_cache(maxsize=512)
def local_geometry(piece: PieceType, mirrored: bool, tol: ToleranceConfig) -> LocalGeometry:
    if piece.category == "straight":
        _require(piece, "length_mm", "width_mm")
        length, width = piece.length_mm, piece.width_mm  # type: ignore[assignment]
        outline = Polygon(
            [(0, -width / 2), (length, -width / 2), (length, width / 2), (0, width / 2)]
        )
        return LocalGeometry(
            piece.slug,
            "straight",
            Pose(0.0, 0.0, 0.0, piece.level_in),
            Pose(length, 0.0, 0.0, piece.level_out),
            length,
            0.0,
            None,
            width,
            outline,
        )
    if piece.category == "curve":
        _require(piece, "turn_angle_deg")
        rc = centerline_radius_mm(piece)
        width = piece.width_mm  # type: ignore[assignment]
        if rc - width / 2 <= 0:
            raise GeometryError(f"piece {piece.slug!r}: inner radius is not positive")
        if mirrored and not piece.mirrorable:
            raise GeometryError(f"piece {piece.slug!r} cannot be mirrored")
        turn = math.radians(piece.turn_angle_deg)  # type: ignore[arg-type]
        sign = -1.0 if mirrored else 1.0
        sweep = sign * turn
        # Left turn: centre at (0, rc); the entry point is at angle -pi/2 seen from the centre.
        centre_y = sign * rc
        start = -sign * math.pi / 2
        outer = _arc_points(rc + width / 2, start, sweep, tol.arc_sagitta_mm)
        inner = _arc_points(rc - width / 2, start, sweep, tol.arc_sagitta_mm)
        ring = [(x, y + centre_y) for x, y in outer] + [
            (x, y + centre_y) for x, y in reversed(inner)
        ]
        exit_pose = Pose(
            rc * math.sin(turn),
            sign * rc * (1 - math.cos(turn)),
            sweep,
            piece.level_out,
        )
        return LocalGeometry(
            piece.slug,
            "curve",
            Pose(0.0, 0.0, 0.0, piece.level_in),
            exit_pose,
            rc * turn,
            sweep,
            rc,
            width,
            Polygon(ring),
        )
    raise GeometryError(
        f"piece {piece.slug!r} (category {piece.category!r}) has no geometry model yet"
    )
