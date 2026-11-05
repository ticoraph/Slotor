"""Rules shared by the full and the incremental validator: one definition of each check (D6)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shapely.geometry import Polygon

from slotor.catalog.bridges import DECK_LEVEL
from slotor.catalog.model import PieceType
from slotor.geometry.tolerance import ToleranceConfig
from slotor.geometry.track import PlacedPiece
from slotor.validation.issues import Issue, IssueCode, Surface


def piece_issues(piece: PieceType, index: int) -> list[Issue]:
    if piece.verified:
        return []
    return [
        Issue(
            IssueCode.UNVERIFIED_PIECE,
            f"piece {piece.slug!r} is not verified and cannot be used (D5)",
            (index,),
        )
    ]


def inventory_issues(used: Mapping[str, int], inventory: Mapping[str, int]) -> list[Issue]:
    """Counts above the stock. Reserved ``@`` entries (``@elevated``) are unlimited when absent."""
    issues = []
    for key, count in sorted(used.items()):
        if key.startswith("@") and key not in inventory:
            continue
        if count > inventory.get(key, 0):
            issues.append(
                Issue(
                    IssueCode.INVENTORY_EXCEEDED,
                    f"{key!r}: {count} used, {inventory.get(key, 0)} available",
                )
            )
    return issues


def surface_issues(
    placed: PlacedPiece, surface: Surface | None, tol: ToleranceConfig
) -> list[Issue]:
    if surface is None:
        return []
    minx, miny, maxx, maxy = placed.outline.bounds
    left, bottom, right, top = surface.bounds
    slack = tol.surface_tol_mm
    inside = (
        minx >= left - slack
        and miny >= bottom - slack
        and maxx <= right + slack
        and maxy <= top + slack
    )
    if inside:
        return []
    return [
        Issue(
            IssueCode.OUT_OF_SURFACE,
            f"piece {placed.index} ({placed.geometry.slug!r}) leaves the surface",
            (placed.index,),
        )
    ]


@dataclass(frozen=True, slots=True)
class Body:
    """A placed piece reduced to what collision needs: an eroded outline and its levels."""

    index: int
    shape: Polygon
    level_in: int
    level_out: int

    @property
    def is_full(self) -> bool:
        return self.level_in == self.level_out


def make_body(placed: PlacedPiece, tol: ToleranceConfig) -> Body:
    """Erode the outline by half the tolerated penetration depth (D24).

    Neighbours sharing a face, and rails side by side, then stop overlapping; only a real
    penetration deeper than ``collision_depth_mm`` survives. Mitred joins keep corners sharp.
    """
    shape = placed.outline.buffer(-tol.collision_depth_mm / 2, join_style="mitre")
    return Body(placed.index, shape, placed.entry.level, placed.exit.level)


def bodies_collide(a: Body, b: Body, tol: ToleranceConfig) -> bool:
    """Area test (never ``overlaps``/``intersects`` alone), with the two-level exception (D4, D30).

    An overlap is tolerated only between two *full* pieces (level_in == level_out) at least
    ``DECK_LEVEL`` apart: a deck over the ground. A ramp element is on the slope, without enough
    room underneath or above, so any overlap with it collides.
    """
    if a.is_full and b.is_full and abs(a.level_in - b.level_in) >= DECK_LEVEL:
        return False
    if a.shape.is_empty or b.shape.is_empty:
        return False
    return a.shape.intersection(b.shape).area > tol.collision_area_mm2
