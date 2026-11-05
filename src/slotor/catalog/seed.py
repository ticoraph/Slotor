"""Read, validate and import the versioned JSON seed of the catalog (D5, D18)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from importlib import resources
from pathlib import Path
from typing import Any

SEED_NAME = "carrera_d132.seed.json"

# Fields a verified piece must carry before an optimizer may use it (D5, D20).
_REQUIRED_WHEN_VERIFIED: dict[str, tuple[str, ...]] = {
    "straight": ("length_mm", "width_mm"),
    "curve": ("radius_mm", "radius_ref", "turn_angle_deg", "width_mm"),
    "banked_curve": ("radius_mm", "radius_ref", "turn_angle_deg", "width_mm"),
}

_PIECE_COLUMNS = (
    "slug",
    "category",
    "name",
    "ta_code",
    "ta_label",
    "manufacturer_part_number",
    "length_mm",
    "radius_mm",
    "radius_ref",
    "turn_angle_deg",
    "mirrorable",
    "width_mm",
    "level_in",
    "level_out",
    "verified",
    "note",
)


class SeedError(ValueError):
    """The seed file is inconsistent or conflicts with the database."""


def seed_bytes(path: Path | None = None) -> bytes:
    if path is not None:
        return path.read_bytes()
    return resources.files("slotor.catalog").joinpath("seed", SEED_NAME).read_bytes()


def read_seed(path: Path | None = None) -> tuple[dict[str, Any], str]:
    """Return ``(parsed seed, sha256 of the raw file)``."""
    raw = seed_bytes(path)
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def validate_seed(data: dict[str, Any]) -> None:
    """Raise :class:`SeedError` on any inconsistency. Never repairs or guesses a value."""
    source_keys = [s["key"] for s in data["sources"]]
    if len(source_keys) != len(set(source_keys)):
        raise SeedError("duplicate source key")

    slugs: set[str] = set()
    for piece in data["pieces"]:
        slug = piece["slug"]
        if slug in slugs:
            raise SeedError(f"duplicate piece slug {slug!r}")
        slugs.add(slug)
        if piece.get("source") not in source_keys:
            raise SeedError(f"piece {slug!r}: unknown source {piece.get('source')!r}")
        if piece.get("verified", False):
            required = _REQUIRED_WHEN_VERIFIED.get(piece["category"])
            if required is None:
                raise SeedError(
                    f"piece {slug!r}: category {piece['category']!r} cannot be verified yet"
                )
            missing = [f for f in required if piece.get(f) is None]
            if missing:
                raise SeedError(f"verified piece {slug!r} lacks {', '.join(missing)}")

    refs: set[str] = set()
    for pack in data["packs"]:
        ref = pack["ref"]
        if len(ref) != 5 or not ref.isdigit():
            raise SeedError(f"pack ref {ref!r} must be 5 digits")
        if ref in refs:
            raise SeedError(f"duplicate pack ref {ref!r}")
        refs.add(ref)
        for key in ("items", "covers"):
            for item in pack.get(key, []):
                if item["slug"] not in slugs:
                    raise SeedError(f"pack {ref}: unknown piece slug {item['slug']!r}")


def import_seed(conn: sqlite3.Connection, path: Path | None = None) -> int:
    """Import the seed as a catalog version and return its id.

    Re-importing identical content is a no-op. Same version name with different content is refused:
    bump ``catalog_version.name`` in the seed instead.
    """
    data, digest = read_seed(path)
    validate_seed(data)
    version = data["catalog_version"]

    existing = conn.execute(
        "SELECT id, content_sha256 FROM catalog_version WHERE name = ?", (version["name"],)
    ).fetchone()
    if existing is not None:
        if existing["content_sha256"] != digest:
            raise SeedError(
                f"catalog version {version['name']!r} already imported with different content; "
                "bump catalog_version.name"
            )
        return int(existing["id"])

    with conn:
        source_ids: dict[str, int] = {}
        for src in data["sources"]:
            conn.execute(
                "INSERT INTO source(name, kind, url, retrieved_on, note) VALUES (?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET kind=excluded.kind, url=excluded.url, "
                "retrieved_on=excluded.retrieved_on, note=excluded.note",
                (
                    src["name"],
                    src["kind"],
                    src.get("url"),
                    src.get("retrieved_on"),
                    src.get("note"),
                ),
            )
            source_ids[src["key"]] = conn.execute(
                "SELECT id FROM source WHERE name = ?", (src["name"],)
            ).fetchone()["id"]

        cur = conn.execute(
            "INSERT INTO catalog_version(name, product_line, content_sha256, note) "
            "VALUES (?,?,?,?)",
            (version["name"], version["product_line"], digest, version.get("note")),
        )
        version_id = int(cur.lastrowid)

        piece_ids: dict[str, int] = {}
        columns = ", ".join(("catalog_version_id", "source_id", *_PIECE_COLUMNS))
        marks = ", ".join("?" * (len(_PIECE_COLUMNS) + 2))
        for piece in data["pieces"]:
            values = [
                version_id,
                source_ids[piece["source"]],
                *(_column_value(piece, col) for col in _PIECE_COLUMNS),
            ]
            cur = conn.execute(f"INSERT INTO piece_type({columns}) VALUES ({marks})", values)
            piece_ids[piece["slug"]] = int(cur.lastrowid)

        # Shoulder packs are not modelled in V1 (D20): only track packs become piece_set rows.
        for pack in data["packs"]:
            if pack["kind"] != "track":
                continue
            cur = conn.execute(
                "INSERT INTO piece_set(catalog_version_id, reference, name) VALUES (?,?,?)",
                (version_id, pack["ref"], pack["name"]),
            )
            for item in pack.get("items", []):
                conn.execute(
                    "INSERT INTO piece_set_item(set_id, piece_type_id, quantity) VALUES (?,?,?)",
                    (cur.lastrowid, piece_ids[item["slug"]], item["qty"]),
                )
    return version_id


def _column_value(piece: dict[str, Any], column: str) -> Any:
    if column in ("mirrorable", "verified"):
        return int(bool(piece.get(column, False)))
    if column in ("level_in", "level_out"):
        return int(piece.get(column, 0))
    return piece.get(column)
