"""Problem and solution types shared by the optimizers, and the only way to build a Solution.

Optimizers search in free space (translation is free) and hand their layouts to ``finalize``,
which places them on the surface and runs the single validator (D6). Only a layout that passes
the full validator becomes a ``Solution``; nothing else ever scores (invariant 1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from slotor.catalog.bridges import variants_of
from slotor.catalog.model import Catalog, PieceType
from slotor.geometry import Pose, ToleranceConfig, build_track
from slotor.geometry.pieces import GeometryError
from slotor.validation import Surface, ValidationResult, validate_layout

SUPPORTED_CATEGORIES = ("straight", "curve")
# The whole layout may be turned by a quarter turn: it then fits the transposed surface.
_ORIENTATIONS_RAD = (0.0, math.pi / 2)


@dataclass(frozen=True)
class Problem:
    """Catalog + inventory + surface. Pieces that cannot be modelled are listed, never used."""

    catalog: Catalog
    inventory: Mapping[str, int]
    surface: Surface
    tol: ToleranceConfig = field(default_factory=ToleranceConfig)

    bridges: bool = False  # allow raised pieces and ramps (D29); otherwise flat only

    @property
    def flat_usable(self) -> tuple[PieceType, ...]:
        """Verified pieces with a geometry model and stock, in a stable (slug) order."""
        pieces = []
        for slug in sorted(self.inventory):
            if self.inventory[slug] <= 0 or slug not in self.catalog:
                continue
            piece = self.catalog.get(slug)
            if piece.verified and piece.category in SUPPORTED_CATEGORIES:
                pieces.append(piece)
        return tuple(pieces)

    @property
    def usable(self) -> tuple[PieceType, ...]:
        """Placeable pieces: the flat ones, plus their level variants when bridges are allowed."""
        flat = self.flat_usable
        if not self.bridges:
            return flat
        extra = [v for piece in flat for v in variants_of(piece)]
        return tuple(sorted([*flat, *extra], key=lambda p: p.slug))

    def piece(self, slug: str) -> PieceType:
        """A catalog piece or one of its level variants (the latter only with bridges)."""
        if slug in self.catalog:
            return self.catalog.get(slug)
        for piece in self.usable:
            if piece.slug == slug:
                return piece
        raise KeyError(slug)

    @property
    def stock(self) -> dict[str, int]:
        """Inventory restricted to the usable pieces, by inventory key (variants share it)."""
        return {p.slug: self.inventory[p.slug] for p in self.flat_usable}

    @property
    def ignored(self) -> dict[str, int]:
        """Inventory the optimizers cannot use (no geometry model yet, or unverified)."""
        usable = {p.slug for p in self.flat_usable}
        return {
            s: n
            for s, n in sorted(self.inventory.items())
            if n > 0 and s not in usable and not s.startswith("@")
        }

    @property
    def inner_size_mm(self) -> tuple[float, float]:
        left, bottom, right, top = self.surface.bounds
        return right - left, top - bottom


@dataclass(frozen=True)
class Solution:
    items: tuple[tuple[PieceType, bool], ...]
    start: Pose
    validation: ValidationResult

    @property
    def length_mm(self) -> float:
        return self.validation.score_mm

    @property
    def n_pieces(self) -> int:
        return len(self.items)

    @property
    def slugs(self) -> tuple[tuple[str, bool], ...]:
        return tuple((piece.slug, mirrored) for piece, mirrored in self.items)


def fits(extent: tuple[float, float], inner: tuple[float, float]) -> bool:
    """Whether a free-space extent fits the inner surface, as is or turned a quarter turn."""
    (w, h), (iw, ih) = extent, inner
    return (w <= iw and h <= ih) or (h <= iw and w <= ih)


def finalize(
    problem: Problem, items: Sequence[tuple[PieceType, bool]], *, require_closed: bool = True
) -> Solution | None:
    """Place ``items`` on the surface and validate. ``None`` when no orientation is valid."""
    if not items:
        return None
    inventory = dict(problem.inventory)
    left, bottom, _, _ = problem.surface.bounds
    for theta in _ORIENTATIONS_RAD:
        try:
            probe = build_track(items, start=Pose(0.0, 0.0, theta, 0), tol=problem.tol)
        except GeometryError:
            return None
        minx, miny, _, _ = probe.bounds
        start = Pose(left - minx, bottom - miny, theta, 0)
        result = validate_layout(
            items,
            inventory,
            surface=problem.surface,
            start=start,
            tol=problem.tol,
            require_closed=require_closed,
        )
        if result.valid:
            return Solution(tuple(items), start, result)
    return None


def make_solution(
    problem: Problem,
    items: Sequence[tuple[PieceType, bool]],
    start: Pose,
    *,
    require_closed: bool = True,
) -> Solution | None:
    """Validate an already placed layout (absolute ``start``) with the full validator."""
    if not items:
        return None
    result = validate_layout(
        items,
        dict(problem.inventory),
        surface=problem.surface,
        start=start,
        tol=problem.tol,
        require_closed=require_closed,
    )
    return Solution(tuple(items), start, result) if result.valid else None
