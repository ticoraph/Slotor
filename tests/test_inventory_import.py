from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slotor.catalog import (
    InventoryParseError,
    load_catalog,
    load_inventory,
    parse_ta_export,
    save_inventory,
)


def test_golden_exports_reproduce_the_captures(catalog, golden_inventories):
    assert len(golden_inventories) >= 2
    for golden in golden_inventories:
        assert parse_ta_export(golden["export_text"], catalog) == golden["counts"], golden["name"]


def test_blank_lines_and_spacing_are_tolerated(catalog):
    text = "\n  3x   Straight, 34,5cm \n\n2x Curve 1/60 degrees, radius 39,6cm\n"
    assert parse_ta_export(text, catalog) == {"straight_345": 3, "curve_r1_60": 2}


def test_repeated_lines_are_summed(catalog):
    assert parse_ta_export("2x Straight, 34,5cm\n3x Straight, 34,5cm", catalog) == {
        "straight_345": 5
    }


def test_unknown_label_is_refused_not_guessed(catalog):
    with pytest.raises(InventoryParseError, match="unknown piece label"):
        parse_ta_export("1x Curve 3/30 degrees, radius 79,2cm", catalog)  # label not observed yet


def test_radius_must_agree_with_catalog(catalog):
    with pytest.raises(InventoryParseError, match="differs from catalog"):
        parse_ta_export("1x Curve 1/60 degrees, radius 40,0cm", catalog)


def test_unparseable_line_is_refused(catalog):
    with pytest.raises(InventoryParseError, match="cannot parse"):
        parse_ta_export("Straight, 34,5cm", catalog)


def test_inventory_roundtrip_through_sqlite(seeded, golden_inventories):
    name = seeded.execute("SELECT name FROM catalog_version").fetchone()["name"]
    catalog = load_catalog(seeded, name)
    golden = golden_inventories[0]
    counts = parse_ta_export(golden["export_text"], catalog)
    save_inventory(seeded, name, golden["name"], counts)
    assert load_inventory(seeded, golden["name"]) == counts


def test_unknown_slug_cannot_be_saved(seeded):
    name = seeded.execute("SELECT name FROM catalog_version").fetchone()["name"]
    with pytest.raises(InventoryParseError, match="unknown piece slug"):
        save_inventory(seeded, name, "bad", {"ghost": 1})
    assert seeded.execute("SELECT COUNT(*) FROM inventory").fetchone()[0] == 0  # rolled back


@settings(deadline=None)
@given(
    st.dictionaries(
        st.sampled_from(
            [
                "curve_r1_60",
                "curve_r2_30",
                "straight_345",
                "straight_115",
                "control_unit",
                "lane_change_double_1",
            ]
        ),
        st.integers(min_value=1, max_value=99),
        min_size=1,
    )
)
def test_format_then_parse_roundtrip(counts):
    from slotor.catalog import import_seed
    from slotor.db import apply_migrations, connect

    conn = connect(":memory:")
    apply_migrations(conn)
    import_seed(conn)
    name = conn.execute("SELECT name FROM catalog_version").fetchone()["name"]
    catalog = load_catalog(conn, name)
    lines = []
    for slug, qty in counts.items():
        piece = catalog.get(slug)
        suffix = f", radius {piece.radius_mm / 10:g}cm".replace(".", ",") if piece.radius_mm else ""
        lines.append(f"{qty}x {piece.ta_label}{suffix}")
    assert parse_ta_export("\n".join(lines), catalog) == counts
    conn.close()
