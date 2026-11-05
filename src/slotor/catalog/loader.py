"""Load a stored catalog version into immutable objects for the core."""

from __future__ import annotations

import sqlite3

from slotor.catalog.model import Catalog, CatalogError, Pack, PieceType

_PIECE_SQL = """
SELECT p.slug, p.category, p.name, p.ta_code, p.ta_label, p.manufacturer_part_number,
       p.length_mm, p.radius_mm, p.radius_ref, p.turn_angle_deg, p.mirrorable, p.width_mm,
       p.level_in, p.level_out, p.verified, s.name AS source, p.note
FROM piece_type p LEFT JOIN source s ON s.id = p.source_id
WHERE p.catalog_version_id = ?
ORDER BY p.id
"""


def load_catalog(conn: sqlite3.Connection, name: str, *, for_optimizer: bool = False) -> Catalog:
    """Load catalog version ``name``.

    With ``for_optimizer=True`` unverified pieces are withheld: ``Catalog.get`` raises
    ``UnverifiedPieceError`` for them (D5). The default view returns everything (UI, import tools).
    """
    version = conn.execute(
        "SELECT id, name, content_sha256 FROM catalog_version WHERE name = ?", (name,)
    ).fetchone()
    if version is None:
        raise CatalogError(f"catalog version {name!r} not found")

    pieces: list[PieceType] = []
    excluded: list[str] = []
    for row in conn.execute(_PIECE_SQL, (version["id"],)):
        piece = PieceType(
            slug=row["slug"],
            category=row["category"],
            name=row["name"],
            ta_code=row["ta_code"],
            ta_label=row["ta_label"],
            manufacturer_part_number=row["manufacturer_part_number"],
            length_mm=row["length_mm"],
            radius_mm=row["radius_mm"],
            radius_ref=row["radius_ref"],
            turn_angle_deg=row["turn_angle_deg"],
            mirrorable=bool(row["mirrorable"]),
            width_mm=row["width_mm"],
            level_in=row["level_in"],
            level_out=row["level_out"],
            verified=bool(row["verified"]),
            source=row["source"],
            note=row["note"],
        )
        if for_optimizer and not piece.verified:
            excluded.append(piece.slug)
        else:
            pieces.append(piece)

    packs: list[Pack] = []
    for pack in conn.execute(
        "SELECT id, reference, name FROM piece_set WHERE catalog_version_id = ? ORDER BY id",
        (version["id"],),
    ):
        items = tuple(
            (item["slug"], item["quantity"])
            for item in conn.execute(
                "SELECT p.slug, i.quantity FROM piece_set_item i "
                "JOIN piece_type p ON p.id = i.piece_type_id WHERE i.set_id = ? ORDER BY p.id",
                (pack["id"],),
            )
        )
        packs.append(Pack(ref=pack["reference"], name=pack["name"], items=items))

    return Catalog(
        name=version["name"],
        content_sha256=version["content_sha256"],
        pieces=tuple(pieces),
        packs=tuple(packs),
        excluded=tuple(excluded),
    )
