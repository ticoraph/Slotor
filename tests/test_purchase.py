"""« Quels rails acheter ? » (phase 8): planner logic with a fake evaluator, then real runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest

from slotor.catalog.bridges import bridge_variants
from slotor.optimizers import Problem
from slotor.purchase import (
    PurchaseJob,
    PurchaseOption,
    make_evaluator,
    plan_purchases,
    purchase_options,
)
from slotor.validation import Surface

from .conftest import cached_catalog


@dataclass
class FakeSolution:
    """Just what the planner reads from a solution."""

    length_mm: float
    items: tuple = field(default_factory=tuple)


def problem(inventory=None) -> Problem:
    return Problem(cached_catalog(), inventory or {"curve_r1_60": 6}, Surface(3000, 2000))


def option(ref: str, price: float, **items: int) -> PurchaseOption:
    return PurchaseOption(ref, f"pack {ref}", tuple(items.items()), price)


def additive(values: dict[str, float], base: float = 1000.0):
    """Length = base + value of each slug bought (units of stock above the starting inventory)."""

    def evaluate(inventory):
        return FakeSolution(base + sum(values.get(s, 0.0) * n for s, n in inventory.items()))

    return evaluate


# --- options -------------------------------------------------------------------------------------


def test_options_are_the_packs_with_a_placeable_content():
    options = {o.ref: o for o in purchase_options(cached_catalog())}
    assert {"20509", "20601", "20571", "20572", "20573", "20577", "20578", "20611", "20612"} <= set(
        options
    )
    assert "20517" not in options and "20574" not in options  # lane change, banked curves
    assert all(o.price == 0 for o in options.values())  # the catalog has no prices (D5)
    assert options["20509"].as_dict() == {"straight_345": 4}


def test_prices_come_from_the_user_only():
    options = {o.ref: o for o in purchase_options(cached_catalog(), {"20509": 19.9})}
    assert options["20509"].price == 19.9 and options["20601"].price == 0


# --- planner logic (fake evaluator) -------------------------------------------------------------


def test_nothing_is_bought_without_prices_or_budget():
    ev = additive({"straight_345": 100})
    free = [option("A", 0, straight_345=2)]
    assert plan_purchases(problem(), free, 100, evaluator=ev).best.steps == ()
    assert (
        plan_purchases(problem(), [option("A", 10, straight_345=2)], 5, evaluator=ev).best.steps
        == ()
    )
    assert (
        plan_purchases(problem(), [option("A", 10, straight_345=2)], 0, evaluator=ev).best.steps
        == ()
    )


def test_the_budget_is_never_exceeded_and_buying_stops_when_nothing_helps():
    ev = additive({"straight_345": 100}, base=0.0)
    result = plan_purchases(problem(), [option("A", 10, straight_345=1)], 35, evaluator=ev)
    plan = result.best
    assert plan.packs == {"A": 3} and plan.pack_cost == 30 and plan.total_cost <= 35
    assert plan.length_mm - plan.baseline_length_mm == pytest.approx(300.0)


def test_a_pack_that_adds_nothing_is_not_bought():
    ev = additive({"straight_345": 100})
    options = [option("A", 5, curve_r4_15=12), option("B", 10, straight_345=1)]
    plan = plan_purchases(problem(), options, 100, evaluator=ev).best
    assert set(plan.packs) == {"B"}


def test_ratio_and_gain_strategies_can_disagree_and_the_better_plan_wins():
    # A: +100 for 10 (ratio 10).  B: +500 for 100 (ratio 5).  Budget 100 buys A once or B once.
    def ev(inventory):  # A is worth 100 once, B 500 once
        has = lambda slug: inventory.get(slug, 0) > 0  # noqa: E731
        return FakeSolution(100.0 * has("straight_345") + 500.0 * has("curve_r1_30"))

    options = [option("A", 10, straight_345=1), option("B", 100, curve_r1_30=1)]
    result = plan_purchases(problem(), options, 100, evaluator=ev)
    by_strategy = {p.strategy: p for p in result.plans}
    assert by_strategy["gain"].packs == {"B": 1} and by_strategy["gain"].length_mm == 500
    assert by_strategy["ratio"].length_mm == 100  # spent on A first, B no longer affordable
    assert result.best.strategy == "gain"


def test_a_pair_is_bought_when_no_single_pack_helps():
    def synergy(inventory):
        both = inventory.get("straight_345", 0) > 0 and inventory.get("curve_r2_30", 0) > 0
        return FakeSolution(2000.0 if both else 1000.0)

    options = [option("S", 10, straight_345=1), option("C", 10, curve_r2_30=1)]
    plan = plan_purchases(problem(), options, 25, evaluator=synergy).best
    assert plan.packs == {"S": 1, "C": 1}
    assert (
        len(plan.steps) == 1
        and set(plan.steps[0].refs) == {"S", "C"}
        and plan.steps[0].gain_mm == 1000
    )


def test_supports_are_part_of_the_cost_unless_already_owned():
    variant = bridge_variants(cached_catalog())["straight_345@deck"]
    bridged = FakeSolution(5000.0, items=((variant, False),) * 7)

    def evaluate(inventory):
        return bridged if inventory.get("straight_345", 0) > 0 else FakeSolution(1000.0)

    options = [option("A", 10, straight_345=1)]
    paid = plan_purchases(problem(), options, 100, support_price=2.0, evaluator=evaluate).best
    assert paid.support_cost == 14.0 and paid.total_cost == 24.0
    tight = plan_purchases(problem(), options, 20, support_price=2.0, evaluator=evaluate).best
    assert tight.steps == ()  # 10 + 7 * 2 = 24 > 20
    owned = problem({"curve_r1_60": 6, "@elevated": 7})
    free = plan_purchases(owned, options, 20, support_price=2.0, evaluator=evaluate).best
    assert free.support_cost == 0.0 and free.packs == {"A": 1}


def test_progress_is_reported_once_per_purchase_step():
    seen = []
    ev = additive({"straight_345": 100}, base=0.0)
    plan_purchases(
        problem(), [option("A", 10, straight_345=1)], 30, evaluator=ev, on_progress=seen.append
    )
    assert [len(p.steps) for p in seen if p.strategy == "ratio"] == [1, 2, 3]


def test_cancellation_and_time_budget_return_what_is_known():
    ev = additive({"straight_345": 100})
    cancelled = plan_purchases(
        problem(), [option("A", 10, straight_345=1)], 100, evaluator=ev, should_stop=lambda: True
    )
    assert cancelled.stop_reason == "cancelled" and cancelled.best.steps == ()

    def slow(inventory):
        time.sleep(0.05)
        return FakeSolution(1000.0 + 100 * inventory.get("straight_345", 0))

    timed = plan_purchases(
        problem(), [option("A", 1, straight_345=1)], 1000, evaluator=slow, time_budget_s=0.3
    )
    assert timed.stop_reason == "time" and len(timed.best.steps) < 1000


def test_invalid_arguments_are_refused():
    with pytest.raises(ValueError, match="budget"):
        plan_purchases(problem(), [], -1, evaluator=additive({}))
    with pytest.raises(ValueError, match="support price"):
        plan_purchases(problem(), [], 1, support_price=-1, evaluator=additive({}))


# --- real optimizers -----------------------------------------------------------------------------


def oval_problem(**kwargs) -> Problem:
    return Problem(
        cached_catalog(), {"curve_r1_60": 6, "straight_345": 4}, Surface(2600, 900), **kwargs
    )


STRAIGHT_PACKS = [
    PurchaseOption("20509", "4 straights", (("straight_345", 4),), 20.0),
    PurchaseOption("20601", "2 straights", (("straight_345", 2),), 11.0),
]


def test_the_evaluator_is_deterministic_and_memoised():
    evaluate = make_evaluator(oval_problem(), seed=0, effort="fast")
    first = evaluate({"curve_r1_60": 6, "straight_345": 4})
    second = evaluate({"straight_345": 4, "curve_r1_60": 6})
    assert first is second  # same inventory, same cache entry
    other = make_evaluator(oval_problem(), seed=0, effort="fast")(
        {"curve_r1_60": 6, "straight_345": 4}
    )
    assert other.length_mm == pytest.approx(first.length_mm)


def test_buying_straights_lengthens_the_oval_by_exactly_their_length():
    result = plan_purchases(oval_problem(), STRAIGHT_PACKS, 20, effort="fast")
    assert result.baseline.length_mm == pytest.approx(3246.106036232337)
    assert result.best.packs == {"20509": 1}
    assert result.best.gain_mm == pytest.approx(4 * 345.0)
    assert result.best.solution.validation.valid
    cheap = plan_purchases(oval_problem(), STRAIGHT_PACKS, 11, effort="fast").best
    assert cheap.packs == {"20601": 1} and cheap.gain_mm == pytest.approx(2 * 345.0)
    both = plan_purchases(oval_problem(), STRAIGHT_PACKS, 31, effort="fast").best
    assert both.pack_cost <= 31 and both.gain_mm == pytest.approx(6 * 345.0)


def test_no_purchase_when_the_surface_cannot_use_more_pieces():
    tiny = Problem(cached_catalog(), {"curve_r1_60": 6, "straight_345": 4}, Surface(1500, 800))
    result = plan_purchases(tiny, STRAIGHT_PACKS, 100, effort="fast")
    assert result.best.steps == () and result.best.gain_mm == 0.0


def test_the_plan_is_reproducible():
    one = plan_purchases(oval_problem(), STRAIGHT_PACKS, 31, effort="fast", seed=3)
    two = plan_purchases(oval_problem(), STRAIGHT_PACKS, 31, effort="fast", seed=3)
    assert [s.refs for s in one.best.steps] == [s.refs for s in two.best.steps]
    assert one.best.length_mm == pytest.approx(two.best.length_mm)


def test_job_runs_in_the_background_and_can_be_cancelled():
    job = PurchaseJob(oval_problem(), STRAIGHT_PACKS, 31, effort="fast")
    assert job.snapshot().status == "idle"
    job.start()
    job.join(120)
    done = job.snapshot()
    assert done.status == "done" and done.result is not None and done.error is None

    slow = PurchaseJob(oval_problem(), STRAIGHT_PACKS, 1000, effort="thorough")
    slow.start()
    time.sleep(0.5)
    slow.cancel()
    slow.join(120)
    assert slow.snapshot().status == "cancelled"


def test_job_failure_is_reported(monkeypatch):
    monkeypatch.setattr("slotor.purchase.plan_purchases", lambda *a, **k: 1 / 0)
    job = PurchaseJob(oval_problem(), STRAIGHT_PACKS, 10)
    job.start()
    job.join(5)
    assert job.snapshot().status == "failed" and "ZeroDivisionError" in job.snapshot().error


# --- service and presentation --------------------------------------------------------------------


def request(**changes):
    from slotor.service import PurchaseRequest

    base = dict(
        surface_width_mm=2600.0,
        surface_height_mm=900.0,
        margin_mm=0.0,
        inventory={"curve_r1_60": 6, "straight_345": 4},
        prices={"20509": 20.0, "20601": 11.0},
        budget=31.0,
        effort="fast",
    )
    base.update(changes)
    return PurchaseRequest(**base)


def test_price_rows_round_trip_and_blank_means_not_offered():
    from slotor.service import prices_from_rows, purchase_price_rows

    rows = purchase_price_rows(cached_catalog(), {"20509": 19.5})
    by_ref = {row[0]: row for row in rows}
    assert by_ref["20509"][3] == 19.5 and by_ref["20601"][3] is None
    assert "4 ×" in by_ref["20509"][2]
    assert prices_from_rows(rows) == {"20509": 19.5}
    assert prices_from_rows([["20509", "x", "y", ""], ["20601", "x", "y", float("nan")]]) == {}


@pytest.mark.parametrize("bad", [-3, "abc"])
def test_a_bad_price_is_refused(bad):
    from slotor.service import prices_from_rows

    with pytest.raises(ValueError, match="20509"):
        prices_from_rows([["20509", "x", "y", bad]])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prices": {}}, "price"),
        ({"prices": {"20517": 5.0}}, "price|cannot be planned"),
        ({"budget": 0.0}, "budget"),
        ({"effort": "instant"}, "effort"),
        ({"margin_mm": 900.0}, "surface"),
        ({"max_elevated": -2}, "max_elevated"),
    ],
)
def test_invalid_purchase_requests_are_refused(changes, message):
    from slotor.service import build_purchase

    with pytest.raises(ValueError, match=message):
        build_purchase(cached_catalog(), request(**changes))


def test_build_purchase_applies_the_bridge_options():
    from slotor.service import build_purchase

    problem, options = build_purchase(cached_catalog(), request(allow_bridges=True, max_elevated=7))
    assert problem.bridges and problem.inventory["@elevated"] == 7
    assert {o.ref for o in options if o.price > 0} == {"20509", "20601"}


def test_the_presentation_lists_steps_a_shopping_list_and_before_after_drawings():
    from slotor.service import build_purchase, present_purchase

    req = request()
    problem, options = build_purchase(cached_catalog(), req)
    job = PurchaseJob(problem, options, req.budget, effort="fast")
    job.start()
    job.join(120)
    shown = present_purchase(problem, job.snapshot(), req)
    assert shown.status.startswith("Terminé")
    assert shown.steps and shown.steps[0][0] == 1
    assert "20509" in shown.shopping_list and "Total : 31" in shown.shopping_list
    assert shown.drawings_html.count("<svg") == 2
    assert "Avant achat" in shown.drawings_html and "Après achat" in shown.drawings_html
    assert "+" in shown.summary and "m" in shown.summary
    assert "straight" in shown.bom.lower()


def test_the_presentation_says_so_when_nothing_helps():
    from slotor.service import build_purchase, present_purchase

    req = request(surface_width_mm=1500.0, surface_height_mm=800.0)
    problem, options = build_purchase(cached_catalog(), req)
    job = PurchaseJob(problem, options, req.budget, effort="fast")
    job.start()
    job.join(120)
    shown = present_purchase(problem, job.snapshot(), req)
    assert "Aucun achat" in shown.summary and shown.steps == []


def test_a_failed_job_is_presented_as_a_failure():
    from slotor.purchase import PurchaseSnapshot
    from slotor.service import present_purchase

    snapshot = PurchaseSnapshot("failed", 1.0, None, None, "RuntimeError: boom")
    assert (
        present_purchase(oval_problem(), snapshot, request()).status == "Échec : RuntimeError: boom"
    )
