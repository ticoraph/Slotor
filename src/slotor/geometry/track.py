"""A chain of placed pieces: poses, closure, total turn, bounds and lane lengths."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from shapely import affinity
from shapely.geometry import Polygon

from slotor.catalog.model import PieceType
from slotor.geometry.pieces import GeometryError, LevelMismatchError, LocalGeometry, local_geometry
from slotor.geometry.pose import Pose, wrap_angle
from slotor.geometry.tolerance import ToleranceConfig


@dataclass(frozen=True, slots=True)
class ClosureError:
    dx: float
    dy: float
    dtheta: float  # wrapped into (-pi, pi]
    dlevel: int

    @property
    def distance_mm(self) -> float:
        return math.hypot(self.dx, self.dy)


@dataclass(frozen=True, slots=True)
class PlacedPiece:
    index: int
    geometry: LocalGeometry
    entry: Pose
    exit: Pose
    outline: Polygon  # in the track frame


@dataclass(frozen=True)
class Track:
    start: Pose
    pieces: tuple[PlacedPiece, ...]

    @property
    def exit_pose(self) -> Pose:
        return self.pieces[-1].exit if self.pieces else self.start

    @property
    def length_mm(self) -> float:
        """Centerline length (D7); analytic, never a polygon approximation."""
        return math.fsum(p.geometry.length_mm for p in self.pieces)

    @property
    def total_turn_deg(self) -> float:
        return math.degrees(math.fsum(p.geometry.turn_rad for p in self.pieces))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        if not self.pieces:
            raise GeometryError("an empty track has no bounds")
        boxes = [p.outline.bounds for p in self.pieces]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def closure_error(self) -> ClosureError:
        end = self.exit_pose
        return ClosureError(
            end.x - self.start.x,
            end.y - self.start.y,
            wrap_angle(end.theta - self.start.theta),
            end.level - self.start.level,
        )

    def is_closed(self, tol: ToleranceConfig) -> bool:
        if not self.pieces:
            return False
        err = self.closure_error()
        # Angular play is a displacement at the rail edge: angle * width <= closure_angle_edge_mm.
        edge_gap_mm = abs(err.dtheta) * self.pieces[0].geometry.width_mm
        return (
            err.distance_mm <= tol.closure_pos_mm
            and edge_gap_mm <= tol.closure_angle_edge_mm
            and err.dlevel == 0
        )

    def lane_length_mm(self, offset_mm: float) -> float:
        """Length of a line parallel to the centerline, ``offset_mm`` to the left (negative: right).

        The lane spacing is not in the catalog yet (measurement sheet A2): the caller supplies it.
        """
        total = 0.0
        for piece in self.pieces:
            geom = piece.geometry
            if geom.kind == "straight":
                total += geom.length_mm
                continue
            radius = geom.centerline_radius_mm - offset_mm * math.copysign(1.0, geom.turn_rad)  # type: ignore[operator]
            if radius <= 0:
                raise GeometryError(
                    f"lane offset {offset_mm} mm exceeds the radius of {geom.slug!r}"
                )
            total += radius * abs(geom.turn_rad)
        return total


def place_piece(
    pose: Pose, piece: PieceType, mirrored: bool, tol: ToleranceConfig, index: int = 0
) -> PlacedPiece:
    """Attach ``piece`` at ``pose``, the end of the chain (shared by full and incremental paths)."""
    geom = local_geometry(piece, mirrored, tol)
    if geom.entry.level != pose.level:
        raise LevelMismatchError(
            f"piece {piece.slug!r} at index {index} enters at level {geom.entry.level}, "
            f"track end is at level {pose.level}"
        )
    outline = affinity.rotate(geom.outline, pose.theta, origin=(0, 0), use_radians=True)
    outline = affinity.translate(outline, pose.x, pose.y)
    return PlacedPiece(index, geom, pose, pose.compose(geom.exit), outline)


def build_track(
    items: Sequence[tuple[PieceType, bool]],
    start: Pose | None = None,
    tol: ToleranceConfig | None = None,
) -> Track:
    """Chain ``(piece, mirrored)`` items end to end from ``start``."""
    start = Pose(0.0, 0.0, 0.0, 0) if start is None else start
    tol = ToleranceConfig() if tol is None else tol
    placed: list[PlacedPiece] = []
    pose = start
    for index, (piece, mirrored) in enumerate(items):
        item = place_piece(pose, piece, mirrored, tol, index)
        placed.append(item)
        pose = item.exit
    return Track(start, tuple(placed))
