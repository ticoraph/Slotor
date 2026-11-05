"""Import a Track Architect export text into an internal inventory.

Export lines look like ``13x Curve 1/60 degrees, radius 39,6cm`` or ``9x Straight, 34,5cm``.
A line is matched on its label (the text before ``, radius ...``) against ``PieceType.ta_label``.
Unknown labels are refused, never guessed.
When the line carries a radius it must agree with the catalog.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter

from slotor.catalog.model import Catalog


class InventoryParseError(ValueError):
    """An export line could not be matched to a catalog piece."""


_LINE = re.compile(
    r"^\s*(?P<qty>\d+)\s*x\s+(?P<label>.+?)"
    r"(?:\s*,\s*radius\s+(?P<radius>\d+(?:[.,]\d+)?)\s*cm)?\s*$",
    re.IGNORECASE,
)
_CM_TO_MM = 10.0


def _normalize(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip()).lower()


def parse_ta_export(text: str, catalog: Catalog, *, radius_tol_mm: float = 1.0) -> dict[str, int]:
    """Return ``{slug: quantity}``. Repeated lines for the same piece are summed.

    ``radius_tol_mm`` covers the export's rounding (radii are printed in cm with one decimal).
    """
    by_label = {_normalize(p.ta_label): p for p in catalog.pieces if p.ta_label}
    counts: Counter[str] = Counter()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _LINE.match(line)
        if match is None:
            raise InventoryParseError(f"line {number}: cannot parse {line.strip()!r}")
        label = match["label"]
        piece = by_label.get(_normalize(label))
        if piece is None:
            raise InventoryParseError(f"line {number}: unknown piece label {label!r}")
        if match["radius"] is not None:
            if piece.radius_mm is None:
                raise InventoryParseError(
                    f"line {number}: export gives a radius but the catalog has none "
                    f"for {piece.slug!r}"
                )
            exported_mm = float(match["radius"].replace(",", ".")) * _CM_TO_MM
            if abs(exported_mm - piece.radius_mm) > radius_tol_mm:
                raise InventoryParseError(
                    f"line {number}: radius {exported_mm:g} mm differs from catalog "
                    f"{piece.radius_mm:g} mm for {piece.slug!r}"
                )
        counts[piece.slug] += int(match["qty"])
    return dict(counts)


def save_inventory(
    conn: sqlite3.Connection,
    catalog_name: str,
    inventory_name: str,
    counts: dict[str, int],
    note: str | None = None,
) -> int:
    """Store an inventory against a catalog version and return its id."""
    version = conn.execute(
        "SELECT id FROM catalog_version WHERE name = ?", (catalog_name,)
    ).fetchone()
    if version is None:
        raise InventoryParseError(f"catalog version {catalog_name!r} not found")
    with conn:
        cur = conn.execute(
            "INSERT INTO inventory(catalog_version_id, name, note) VALUES (?,?,?)",
            (version["id"], inventory_name, note),
        )
        for slug, quantity in counts.items():
            piece = conn.execute(
                "SELECT id FROM piece_type WHERE catalog_version_id = ? AND slug = ?",
                (version["id"], slug),
            ).fetchone()
            if piece is None:
                raise InventoryParseError(f"unknown piece slug {slug!r}")
            conn.execute(
                "INSERT INTO inventory_item(inventory_id, piece_type_id, quantity) VALUES (?,?,?)",
                (cur.lastrowid, piece["id"], quantity),
            )
    return int(cur.lastrowid)


def load_inventory(conn: sqlite3.Connection, inventory_name: str) -> dict[str, int]:
    rows = conn.execute(
        "SELECT p.slug, i.quantity FROM inventory inv "
        "JOIN inventory_item i ON i.inventory_id = inv.id "
        "JOIN piece_type p ON p.id = i.piece_type_id WHERE inv.name = ? ORDER BY p.id",
        (inventory_name,),
    ).fetchall()
    return {row["slug"]: row["quantity"] for row in rows}
