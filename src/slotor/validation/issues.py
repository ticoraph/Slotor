"""Validation vocabulary: issue codes, issues, results and the work surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from slotor.geometry.track import Track


class IssueCode(StrEnum):
    EMPTY = "empty"
    UNVERIFIED_PIECE = "unverified_piece"
    INVENTORY_EXCEEDED = "inventory_exceeded"
    GEOMETRY = "geometry"
    LEVEL_MISMATCH = "level_mismatch"
    OUT_OF_SURFACE = "out_of_surface"
    NOT_CLOSED = "not_closed"
    NOT_AT_GROUND = "not_at_ground"
    COLLISION = "collision"


@dataclass(frozen=True, slots=True)
class Issue:
    code: IssueCode
    message: str
    pieces: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Surface:
    """Rectangle ``[0, width] x [0, height]`` (mm); pieces must stay ``margin_mm`` inside it."""

    width_mm: float
    height_mm: float
    margin_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.margin_mm < 0 or 2 * self.margin_mm >= min(self.width_mm, self.height_mm):
            raise ValueError("surface margin leaves no usable area")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        m = self.margin_mm
        return (m, m, self.width_mm - m, self.height_mm - m)


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[Issue, ...]
    track: Track | None = None

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def codes(self) -> frozenset[IssueCode]:
        return frozenset(issue.code for issue in self.issues)

    @property
    def score_mm(self) -> float:
        """Centerline length (D7) of a valid layout; an invalid one never scores (invariant 1)."""
        return self.track.length_mm if self.valid and self.track is not None else 0.0
