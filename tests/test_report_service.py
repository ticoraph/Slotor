"""Report exports and the application service (Phase 5): the logic the UI relies on."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET

import pytest

from slotor.optimizers import generate_seeds
from slotor.report import (
    bill_of_materials,
    build_report,
    format_summary,
    solution_from_json,
    solution_to_json,
    solution_to_svg,
    usage_rows,
)
from slotor.service import (
    OptimizationJob,
    OptimizeRequest,
    build_problem,
    default_catalog,
    import_ta_export,
    inventory_from_rows,
    inventory_rows,
    optimize,
)

from .test_optimizers import problem


@pytest.fixture(scope="module")
def catalog():
    return default_catalog()


# --- inventory input ---------------------------------------------------------------------------


def test_inventory_table_lists_only_optimizable_pieces(catalog):
    rows = inventory_rows(catalog)
    slugs = {row[0] for row in rows}
    assert {"straight_345", "curve_r1_60"} <= slugs
    assert "control_unit" not in slugs and "bridge_deck" not in slugs
    assert all(row[2] == 0 for row in rows)


def test_rows_round_trip_through_the_table(catalog):
    inventory = {"curve_r1_60": 6, "straight_345": 4}
    assert inventory_from_rows(catalog, inventory_rows(catalog, inventory)) == inventory


@pytest.mark.parametrize("bad", [-1, 1.5, "abc"])
def test_bad_quantity_is_refused(catalog, bad):
    with pytest.raises(ValueError, match="straight_345"):
        inventory_from_rows(catalog, [["straight_345", "x", bad]])


def test_blank_and_nan_quantities_count_as_zero(catalog):
    rows = [["straight_345", "x", None], ["curve_r1_60", "y", float("nan")], ["", "", 3]]
    assert inventory_from_rows(catalog, rows) == {}


def test_unknown_piece_in_a_row_is_refused(catalog):
    with pytest.raises(ValueError, match="unknown piece"):
        inventory_from_rows(catalog, [["warp_drive", "x", 1]])


def test_track_architect_export_import(catalog):
    text = "13x Curve 1/60 degrees, radius 39,6cm\n9x Straight, 34,5cm\n1x Control unit"
    assert import_ta_export(catalog, text) == {
        "curve_r1_60": 13,
        "straight_345": 9,
        "control_unit": 1,
    }
    with pytest.raises(ValueError):
        import_ta_export(catalog, "3x Flux capacitor")


# --- problem building --------------------------------------------------------------------------


def request(**changes) -> OptimizeRequest:
    base = dict(
        surface_width_mm=1500.0,
        surface_height_mm=800.0,
        margin_mm=0.0,
        inventory={"curve_r1_60": 6, "straight_345": 4},
    )
    base.update(changes)
    return OptimizeRequest(**base)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"inventory": {}}, "empty"),
        ({"inventory": {"control_unit": 1}}, "no piece"),
        ({"time_budget_s": 0.0}, "time budget"),
        ({"top_k": 0}, "top_k"),
        ({"lane_spacing_mm": -3.0}, "lane spacing"),
        ({"margin_mm": 900.0}, "surface"),
    ],
)
def test_invalid_requests_are_refused(catalog, changes, message):
    with pytest.raises(ValueError, match=message):
        build_problem(catalog, request(**changes))


# --- optimization pipeline ---------------------------------------------------------------------


def test_optimize_returns_distinct_valid_loops_longest_first(catalog):
    prob = problem("kit_mixed")
    result = optimize(prob, seed=0, time_budget_s=3.0, top_k=3)
    assert result.stop_reason == "done" and result.top
    lengths = [s.length_mm for s in result.top]
    assert lengths == sorted(lengths, reverse=True)
    assert len({(round(s.length_mm, 3), s.n_pieces) for s in result.top}) == len(result.top)
    assert lengths[0] > generate_seeds(prob)[0].length_mm
    assert all(s.validation.valid for s in result.top)


def test_optimize_stops_early_when_the_whole_inventory_is_used(catalog):
    prob = problem("kit_oval")
    result = optimize(prob, seed=0, time_budget_s=30.0)
    assert result.elapsed_s < 5.0
    assert result.stop_reason == "done" and "impossible de faire plus long" in result.message
    assert result.best is not None and result.best.n_pieces == 10


def test_optimize_reports_progress_with_strictly_better_solutions(catalog):
    seen: list[tuple[float, str]] = []
    prob = problem("golden_b")
    optimize(
        prob,
        seed=0,
        time_budget_s=3.0,
        top_k=2,
        on_progress=lambda s, st: seen.append((s.length_mm, st)),
    )
    lengths = [length for length, _ in seen]
    assert lengths == sorted(set(lengths))
    assert {stage for _, stage in seen} <= {"seed", "beam", "lns"}


def test_optimize_without_any_loop_says_so(catalog):
    from slotor.optimizers import Problem
    from slotor.validation import Surface

    prob = Problem(catalog, {"curve_r1_60": 2, "straight_345": 3}, Surface(3000, 2000))
    result = optimize(prob, time_budget_s=0.5)
    assert result.best is None and "Aucun circuit" in result.message


def test_job_runs_in_the_background_and_exposes_progress(catalog):
    job = OptimizationJob(problem("golden_b"), time_budget_s=2.0, top_k=2)
    assert job.snapshot().status == "idle"
    job.start()
    assert job.snapshot().status in {"running", "done"}
    job.join(20)
    snapshot = job.snapshot()
    assert snapshot.status == "done" and snapshot.result is not None
    assert snapshot.best is not None and snapshot.error is None


def test_job_can_be_cancelled_and_keeps_its_best(catalog):
    job = OptimizationJob(problem("golden_a"), time_budget_s=60.0, top_k=3)
    job.start()
    time.sleep(1.0)
    job.cancel()
    job.join(20)
    snapshot = job.snapshot()
    assert snapshot.status == "cancelled"
    assert snapshot.elapsed_s < 20.0
    assert snapshot.best is not None and snapshot.best.validation.valid


def test_job_failure_is_reported_not_swallowed(catalog, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("slotor.service.optimize", boom)
    job = OptimizationJob(problem("kit_oval"))
    job.start()
    job.join(5)
    snapshot = job.snapshot()
    assert snapshot.status == "failed" and "kaboom" in (snapshot.error or "")


# --- report and exports ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oval_solution():
    prob = problem("kit_oval")
    return prob, generate_seeds(prob)[0]


def test_report_figures_and_usage(oval_solution):
    prob, solution = oval_solution
    report = build_report(solution, prob)
    assert report.length_mm == pytest.approx(solution.length_mm)
    assert sorted(report.extent_mm) == pytest.approx([792.0, 1482.0], abs=0.5)
    assert {r.slug: (r.used, r.available) for r in report.usage} == {
        "curve_r1_60": (6, 6),
        "straight_345": (4, 4),
    }
    assert report.lane_lengths_mm is None
    assert "entraxe" in format_summary(report)
    assert "10 pièces" in bill_of_materials(report)
    assert usage_rows(report)[0][-1] == 0  # nothing left over


def test_lane_lengths_need_the_spacing_and_follow_the_arc_radius(oval_solution):
    prob, solution = oval_solution
    report = build_report(solution, prob, lane_spacing_mm=60.0)
    left, right = report.lane_lengths_mm
    # A loop made of left-hand turns only: the left lane is the inner one.
    assert left < solution.length_mm < right
    assert right - left == pytest.approx(2 * 30.0 * 2 * 3.141592653589793)
    assert "Voie gauche" in format_summary(report)


def test_unused_stock_and_ignored_pieces_appear_in_the_report():
    prob = problem("golden_a")
    solution = generate_seeds(prob)[0]
    report = build_report(solution, prob)
    assert report.ignored == {
        "control_unit": 1,
        "lane_change_double_1": 2,
        "lane_change_double_2": 2,
    }
    assert "ignorées" in format_summary(report)
    assert any(row.left > 0 for row in report.usage)


def test_json_export_round_trips_through_the_validator(oval_solution):
    prob, solution = oval_solution
    text = solution_to_json(solution, prob, lane_spacing_mm=None)
    data = json.loads(text)
    assert data["format"] == "slotor-solution" and len(data["pieces"]) == 10
    rebuilt = solution_from_json(text, prob)
    assert rebuilt.slugs == solution.slugs
    assert rebuilt.length_mm == pytest.approx(solution.length_mm)


def test_a_tampered_json_file_is_revalidated_and_refused(oval_solution):
    prob, solution = oval_solution
    data = json.loads(solution_to_json(solution, prob))
    data["pieces"].pop()
    with pytest.raises(ValueError, match="not a valid closed loop"):
        solution_from_json(json.dumps(data), prob)
    with pytest.raises(ValueError, match="malformed|not a slotor"):
        solution_from_json("{}", prob)
    with pytest.raises(ValueError, match="malformed"):
        solution_from_json("not json", prob)


def test_svg_export_is_well_formed_and_to_scale(oval_solution):
    prob, solution = oval_solution
    svg = solution_to_svg(solution, prob, width_px=750.0)
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.attrib["width"] == "750"
    assert root.attrib["height"] == str(round(750 * 800 / 1500))
    assert svg.count("<polygon") == solution.n_pieces
    assert "1500 × 800 mm" in svg
