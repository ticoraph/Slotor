"""The Gradio app driven through its public API, no browser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from gradio_client import Client

from slotor import service
from slotor.ui import build_ui

HEADERS = ["Référence", "Pièce", "Quantité"]


@pytest.fixture(scope="module")
def client():
    demo = build_ui().queue()
    _, url, _ = demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", quiet=True)
    try:
        yield Client(url, verbose=False)
    finally:
        demo.close()


def table(inventory: dict[str, int]) -> dict:
    rows = service.inventory_rows(service.default_catalog(), inventory)
    return {"headers": HEADERS, "data": rows, "metadata": None}


def run(
    client, inventory, *, width=1500, height=800, budget="10 s", lane=None, bridges=False, cap=None
):
    return client.predict(
        width, height, 0, budget, 0, 2, lane, bridges, cap, table(inventory), api_name="/on_run"
    )


def test_the_app_exposes_import_run_and_stop(client):
    names = set(client.view_api(return_format="dict")["named_endpoints"])
    assert {"/on_import", "/on_run", "/on_stop"} <= names


def test_running_the_oval_kit_returns_a_drawing_a_parts_list_and_exports(client):
    status, summary, drawings, usage, bom, json_path, svg_path = run(
        client, {"curve_r1_60": 6, "straight_345": 4}
    )
    assert status.startswith("Terminé") and "3.246 m" in status
    assert "3.246 m" in summary and "10 pièces" in summary
    assert drawings.count("<svg") == 1 and "<polygon" in drawings
    assert usage["data"][0][0] == "curve_r1_60" and usage["data"][0][2:] == [6, 6, 0]
    assert "6 × Curve R1 60°" in bom
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert data["format"] == "slotor-solution" and len(data["pieces"]) == 10
    assert Path(svg_path).read_text(encoding="utf-8").startswith("<svg")


def test_the_lane_spacing_field_turns_on_the_two_lane_lengths(client):
    summary = run(client, {"curve_r1_60": 6, "straight_345": 4}, lane=60)[1]
    assert "Voie gauche" in summary and "voie droite" in summary


def test_a_roomy_inventory_gives_several_valid_loops(client):
    inventory = {"curve_r1_60": 12, "straight_345": 16}
    drawings = run(client, inventory, width=2680, height=1840, budget="10 s")[2]
    assert drawings.count("<svg") >= 1
    assert "#1" in drawings


def test_invalid_input_is_reported_without_crashing(client):
    status = run(client, {})[0]
    assert status.startswith("Entrée invalide") and "empty" in status
    status = run(client, {"curve_r1_60": 6, "straight_345": 4}, width=0)[0]
    assert status.startswith("Entrée invalide")


def test_the_track_architect_import_fills_the_table(client):
    table_value, message = client.predict(
        "12x Curve 1/60 degrees, radius 39,6cm\n16x Straight, 34,5cm\n1x Control unit",
        api_name="/on_import",
    )
    quantities = {row[0]: row[2] for row in table_value["data"]}
    assert quantities["curve_r1_60"] == 12 and quantities["straight_345"] == 16
    assert "Ignorées" in message and "control_unit" in message


def test_a_bad_import_leaves_a_message(client):
    _, message = client.predict("3x Flux capacitor", api_name="/on_import")
    assert message.startswith("Import refusé")


def test_stop_without_a_running_job_is_harmless(client):
    client.predict(api_name="/on_stop")


def test_the_server_is_not_exposed_beyond_loopback():
    demo = build_ui()
    _, url, _ = demo.launch(prevent_thread_lock=True, server_name="127.0.0.1", quiet=True)
    try:
        assert url.startswith("http://127.0.0.1")
    finally:
        demo.close()


def test_the_bridge_option_produces_an_actual_bridge(client):
    inventory = {"curve_r1_60": 12, "straight_345": 16}
    status, summary, drawings, usage, bom, json_path, _ = run(
        client, inventory, width=2680, height=1840, budget="10 s", bridges=True, cap=None
    )
    assert status.startswith("Terminé")
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert data["bridges"] is True
    raised = [p["slug"] for p in data["pieces"] if "@" in p["slug"]]
    assert len(raised) >= 7 and "Pont : 1 pont(s)" in summary
    assert "(ramp up 1/3)" in bom and "(raised)" in bom
    assert "#c9a15a" in drawings  # the deck is drawn on top
    assert all(row[2] <= row[3] for row in usage["data"])


def test_the_elevated_cap_is_applied_through_the_interface(client):
    inventory = {"curve_r1_60": 12, "straight_345": 16}
    _, summary, _, _, _, json_path, _ = run(
        client, inventory, width=2680, height=1840, budget="10 s", bridges=True, cap=6
    )
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert data["inventory"]["@elevated"] == 6
    assert not [p for p in data["pieces"] if "@" in p["slug"]]  # no bridge fits under 7
    assert "Pont" not in summary


def test_the_purchase_tab_plans_a_purchase(client):
    prices = service.purchase_price_rows(service.default_catalog(), {"20509": 20.0, "20601": 11.0})
    status, summary, steps, shopping, drawings, bom = client.predict(
        2600,
        900,
        0,
        table({"curve_r1_60": 6, "straight_345": 4}),
        {"headers": ["Lot", "Nom", "Contenu", "Prix"], "data": prices, "metadata": None},
        31,
        0,
        False,
        None,
        "rapide",
        "1 min",
        api_name="/on_buy",
    )
    assert status.startswith("Terminé")
    assert "20509" in shopping and "Total : 31" in shopping
    assert steps["data"] and steps["data"][0][0] == 1
    assert drawings.count("<svg") == 2 and "straight" in bom.lower()
    assert "m" in summary and "dépensés" in summary


def test_the_purchase_tab_reports_missing_prices(client):
    prices = service.purchase_price_rows(service.default_catalog())
    status = client.predict(
        2600,
        900,
        0,
        table({"curve_r1_60": 6}),
        {"headers": ["Lot", "Nom", "Contenu", "Prix"], "data": prices, "metadata": None},
        50,
        0,
        False,
        None,
        "rapide",
        "1 min",
        api_name="/on_buy",
    )[0]
    assert status.startswith("Entrée invalide") and "price" in status
