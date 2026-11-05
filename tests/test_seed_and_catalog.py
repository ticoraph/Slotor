from __future__ import annotations

import copy
import hashlib

import pytest

from slotor.catalog import (
    SeedError,
    UnknownPieceError,
    UnverifiedPieceError,
    import_seed,
    load_catalog,
    read_seed,
    validate_seed,
)
from slotor.catalog.seed import seed_bytes


@pytest.fixture
def seed_data() -> dict:
    return read_seed()[0]


def test_real_seed_is_valid(seed_data):
    validate_seed(seed_data)


def test_seed_has_no_personal_data():
    text = seed_bytes().decode("utf-8")
    assert "/Users/" not in text
    assert "@" not in text  # no e-mail (D17)


def test_every_verified_curve_declares_its_radius_reference(seed_data):
    for piece in seed_data["pieces"]:
        if piece.get("verified") and piece["category"] == "curve":
            assert piece["radius_ref"] == "outer_edge", piece["slug"]  # D22


def test_working_dimensions_match_decision_d21(seed_data):
    pieces = {p["slug"]: p for p in seed_data["pieces"]}
    radii = {s: pieces[s]["radius_mm"] for s in pieces if s.startswith("curve_r")}
    assert radii == {
        "curve_r1_60": 396.0,
        "curve_r1_30": 396.0,
        "curve_r2_30": 594.0,
        "curve_r3_30": 792.0,
        "curve_r4_15": 990.0,
    }
    # Exact nesting with the 198 mm rail (D20/D21).
    width = pieces["curve_r1_60"]["width_mm"]
    assert width == 198.0
    assert radii["curve_r2_30"] - radii["curve_r1_30"] == width
    assert radii["curve_r3_30"] - radii["curve_r2_30"] == width
    assert radii["curve_r4_15"] - radii["curve_r3_30"] == width
    assert pieces["straight_86_25"]["length_mm"] == pieces["straight_345"]["length_mm"] / 4


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["pieces"].append(copy.deepcopy(d["pieces"][0])), "duplicate piece slug"),
        (lambda d: d["pieces"][0].update(source="nope"), "unknown source"),
        (lambda d: d["packs"][0].update(ref="123"), "5 digits"),
        (lambda d: d["packs"].append(copy.deepcopy(d["packs"][0])), "duplicate pack ref"),
        (
            lambda d: d["packs"][0]["items"].append({"slug": "ghost", "qty": 1}),
            "unknown piece slug",
        ),
    ],
)
def test_validate_rejects_inconsistent_seed(seed_data, mutate, message):
    mutate(seed_data)
    with pytest.raises(SeedError, match=message):
        validate_seed(seed_data)


def test_verified_curve_without_radius_reference_is_rejected(seed_data):
    curve = next(p for p in seed_data["pieces"] if p["slug"] == "curve_r2_30")
    curve["radius_ref"] = None
    with pytest.raises(SeedError, match="radius_ref"):
        validate_seed(seed_data)


def test_verified_piece_of_unsupported_category_is_rejected(seed_data):
    lane_change = next(p for p in seed_data["pieces"] if p["slug"] == "lane_change_double_1")
    lane_change["verified"] = True
    with pytest.raises(SeedError, match="cannot be verified"):
        validate_seed(seed_data)


def test_unverified_curve_with_missing_dimensions_is_rejected_once_verified(seed_data):
    hairpin = next(p for p in seed_data["pieces"] if p["slug"] == "hairpin")
    hairpin["verified"] = True
    with pytest.raises(SeedError, match="lacks"):
        validate_seed(seed_data)


def test_import_stores_content_hash(conn):
    import_seed(conn)
    row = conn.execute("SELECT content_sha256 FROM catalog_version").fetchone()
    assert row["content_sha256"] == hashlib.sha256(seed_bytes()).hexdigest()


def test_import_is_idempotent(conn):
    first = import_seed(conn)
    assert import_seed(conn) == first
    assert conn.execute("SELECT COUNT(*) FROM catalog_version").fetchone()[0] == 1


def test_same_version_name_with_different_content_is_refused(conn, tmp_path, seed_data):
    import json

    import_seed(conn)
    seed_data["catalog_version"]["note"] = "changed without bumping the version name"
    altered = tmp_path / "altered.seed.json"
    altered.write_text(json.dumps(seed_data), encoding="utf-8")
    with pytest.raises(SeedError, match="bump"):
        import_seed(conn, altered)


def test_catalog_reloads_identically_from_seed(catalog, seed_data):
    assert [p.slug for p in catalog.pieces] == [p["slug"] for p in seed_data["pieces"]]
    for loaded, raw in zip(catalog.pieces, seed_data["pieces"], strict=True):
        for field in ("category", "name", "length_mm", "radius_mm", "radius_ref", "width_mm"):
            assert getattr(loaded, field) == raw.get(field), (raw["slug"], field)
        assert loaded.turn_angle_deg == raw.get("turn_angle_deg")
        assert loaded.verified is bool(raw.get("verified", False))
        assert loaded.ta_label == raw.get("ta_label")
    assert catalog.content_sha256 == hashlib.sha256(seed_bytes()).hexdigest()


def test_packs_are_imported_as_piece_sets(catalog):
    packs = {p.ref: p for p in catalog.packs}
    assert packs["20509"].items == (("straight_345", 4),)
    assert packs["20577"].items == (("curve_r1_30", 6),)
    assert "20560" not in packs  # shoulder packs are not modelled in V1 (D20)


def test_optimizer_view_refuses_unverified_pieces(seeded):
    name = seeded.execute("SELECT name FROM catalog_version").fetchone()["name"]
    optimizer = load_catalog(seeded, name, for_optimizer=True)
    assert optimizer.get("curve_r1_60").radius_mm == 396.0
    with pytest.raises(UnverifiedPieceError):
        optimizer.get("hairpin")
    with pytest.raises(UnknownPieceError):
        optimizer.get("does_not_exist")
    assert all(p.verified for p in optimizer.pieces)
    assert "hairpin" in optimizer.excluded


def test_full_view_keeps_unverified_pieces(catalog):
    assert not catalog.get("hairpin").verified
    assert catalog.get("curve_r4_15").verified


def test_catalog_objects_are_immutable(catalog):
    piece = catalog.get("straight_345")
    with pytest.raises(AttributeError):
        piece.length_mm = 1.0
