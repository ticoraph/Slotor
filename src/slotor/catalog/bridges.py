"""Bridge pieces (D29, D30): standard straights and curves laid on a staircase of levels.

The user's hardware has no dedicated bridge pieces: a bridge is made of the same straights and
curves, held up by supports bought separately. Reaching the deck takes **three elements** (for
instance three straights, or one straight and two curves), the deck is **one element or more** and
going back down takes three more. Each ramp element therefore changes the level by one step:

    ground 0 -u0-> 1 -u1-> 2 -u2-> 3 (deck, one or more pieces) 3 -d2-> 2 -d1-> 1 -d0-> 0

A variant has exactly the geometry of its base piece (same dimensions, same sources) and consumes
the same inventory entry (``stock_slug``). Slope, height in millimetres, clearance and the supports
themselves are not modelled (D4); the level staircase only encodes who may overlap whom.
"""

from __future__ import annotations

import dataclasses

from slotor.catalog.model import Catalog, PieceType

DECK_LEVEL = 3  # elements needed to climb; a piece may pass over the ground only at this level
DECK = "@deck"
# (suffix, level_in, level_out, label)
_STEPS = (
    ("@u0", 0, 1, "ramp up 1/3"),
    ("@u1", 1, 2, "ramp up 2/3"),
    ("@u2", 2, 3, "ramp up 3/3"),
    (DECK, 3, 3, "raised"),
    ("@d2", 3, 2, "ramp down 1/3"),
    ("@d1", 2, 1, "ramp down 2/3"),
    ("@d0", 1, 0, "ramp down 3/3"),
)
SUPPORTED_CATEGORIES = ("straight", "curve")
# Assumption about the hardware (D30): only full-size elements climb or carry a deck. A 1/3 or
# 1/4 straight is too short to gain height or to span the rail underneath.
MIN_RAISED_STRAIGHT_MM = 345.0


def is_raisable(piece: PieceType) -> bool:
    if not piece.verified or piece.category not in SUPPORTED_CATEGORIES:
        return False
    if piece.category == "straight":
        return (piece.length_mm or 0.0) >= MIN_RAISED_STRAIGHT_MM
    return True


def variants_of(piece: PieceType) -> list[PieceType]:
    """The seven level variants of a raisable piece (empty for the others)."""
    if not is_raisable(piece):
        return []
    note = f"Level variant of {piece.slug} (D30): same geometry, same inventory entry."
    base = dataclasses.replace(piece, stock_slug=piece.slug, note=note)
    return [
        dataclasses.replace(
            base,
            slug=piece.slug + suffix,
            name=f"{piece.name} ({label})",
            level_in=level_in,
            level_out=level_out,
        )
        for suffix, level_in, level_out, label in _STEPS
    ]


def bridge_variants(catalog: Catalog) -> dict[str, PieceType]:
    """Every level variant derivable from the catalog, by slug."""
    result: dict[str, PieceType] = {}
    for piece in catalog.pieces:
        for variant in variants_of(piece):
            result[variant.slug] = variant
    return result
