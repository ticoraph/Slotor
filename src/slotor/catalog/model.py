"""Immutable catalog objects loaded once for the core (D18: no database access in hot loops)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

ELEVATED_KEY = "@elevated"


class CatalogError(Exception):
    """Base class for catalog errors."""


class UnknownPieceError(CatalogError, KeyError):
    """The slug does not exist in the catalog."""


class UnverifiedPieceError(CatalogError, LookupError):
    """The piece exists but is not verified: optimizers must refuse it (D5)."""


@dataclass(frozen=True, slots=True)
class PieceType:
    slug: str
    category: str
    name: str
    ta_code: str | None
    ta_label: str | None
    manufacturer_part_number: str | None
    length_mm: float | None
    radius_mm: float | None
    radius_ref: str | None
    turn_angle_deg: float | None
    mirrorable: bool
    width_mm: float | None
    level_in: int
    level_out: int
    verified: bool
    source: str | None
    note: str | None
    # Derived level variants (bridges, D29) draw on the stock of another piece type.
    stock_slug: str | None = None

    @property
    def inventory_key(self) -> str:
        """The inventory entry this piece consumes (itself, unless it is a level variant)."""
        return self.stock_slug or self.slug

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        """Every inventory entry this piece consumes: its own, plus ``@elevated`` when raised.

        ``@elevated`` is a reserved, optional entry: when absent the number of raised pieces is
        unlimited, when present (even 0) it caps them (supports are bought separately, D29).
        """
        if self.level_in > 0 or self.level_out > 0:
            return (self.inventory_key, ELEVATED_KEY)
        return (self.inventory_key,)


@dataclass(frozen=True, slots=True)
class Pack:
    """Official retail pack (piece_set); ``ref`` is the 5 trailing digits of the item number."""

    ref: str
    name: str | None
    items: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Catalog:
    name: str
    content_sha256: str
    pieces: tuple[PieceType, ...]
    packs: tuple[Pack, ...] = ()
    # Slugs present in the stored catalog but withheld because they are unverified (optimizer view).
    excluded: tuple[str, ...] = ()
    _index: Mapping[str, PieceType] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", MappingProxyType({p.slug: p for p in self.pieces}))

    def get(self, slug: str) -> PieceType:
        piece = self._index.get(slug)
        if piece is not None:
            return piece
        if slug in self.excluded:
            raise UnverifiedPieceError(f"piece {slug!r} is not verified and cannot be used")
        raise UnknownPieceError(slug)

    def __contains__(self, slug: object) -> bool:
        return slug in self._index

    def __iter__(self):
        return iter(self.pieces)
