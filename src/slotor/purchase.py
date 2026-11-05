"""« Quels rails acheter ? » (phase 8, D32): the longest loop a purchase budget can buy.

The catalog has no prices (D5): the user supplies a price for each pack they would consider, and a
budget in the same unit. Candidates are the official packs whose content the optimizers can place
(``Pack.items``). The planner is a greedy search around the existing optimizers: each step adds the
pack with the best gain, scored two ways (gain per unit of money, and plain gain), falling back to
pairs of packs when no single pack helps (a stadium needs curves *and* straights). Every length
comes from a validated loop; nothing here changes how a loop is checked.
"""

from __future__ import annotations

import dataclasses
import itertools
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from slotor.catalog.model import ELEVATED_KEY, Catalog
from slotor.optimizers import (
    BeamConfig,
    LNSConfig,
    Problem,
    Solution,
    beam_search,
    bridge_seeds,
    generate_seeds,
    lns,
)

EPSILON_MM = 1.0  # a purchase must add at least this much length to count
EFFORTS = {"fast": (3000, 25), "normal": (12000, 70), "thorough": (30000, 160)}  # beam, LNS
Evaluator = Callable[[Mapping[str, int]], "Solution | None"]


@dataclass(frozen=True)
class PurchaseOption:
    ref: str  # 5 trailing digits of the official article number
    name: str
    items: tuple[tuple[str, int], ...]
    price: float = 0.0

    def as_dict(self) -> dict[str, int]:
        return dict(self.items)


def purchase_options(
    catalog: Catalog, prices: Mapping[str, float] | None = None
) -> list[PurchaseOption]:
    """Packs whose whole content is placeable, with the user's price (0 = not offered)."""
    probe = Problem(catalog, {p.slug: 1 for p in catalog.pieces}, _unit_surface())
    placeable = {p.slug for p in probe.flat_usable}
    prices = prices or {}
    options = []
    for pack in catalog.packs:
        if pack.items and all(slug in placeable for slug, _ in pack.items):
            options.append(
                PurchaseOption(
                    pack.ref, pack.name or pack.ref, pack.items, float(prices.get(pack.ref, 0))
                )
            )
    return sorted(options, key=lambda o: o.ref)


def _unit_surface():
    from slotor.validation import Surface

    return Surface(1000.0, 1000.0)


@dataclass(frozen=True)
class PurchaseStep:
    refs: tuple[str, ...]  # one pack, or two when a pair was needed
    names: tuple[str, ...]
    price: float
    cumulative_price: float
    length_mm: float
    gain_mm: float


@dataclass(frozen=True)
class PurchasePlan:
    strategy: str  # "ratio" (gain per unit of money) or "gain"
    steps: tuple[PurchaseStep, ...]
    packs: dict[str, int]  # ref -> number bought
    inventory: dict[str, int]  # inventory after buying
    solution: Solution | None
    pack_cost: float
    support_cost: float
    baseline_length_mm: float

    @property
    def total_cost(self) -> float:
        return self.pack_cost + self.support_cost

    @property
    def length_mm(self) -> float:
        return self.solution.length_mm if self.solution is not None else 0.0

    @property
    def gain_mm(self) -> float:
        return self.length_mm - self.baseline_length_mm


@dataclass(frozen=True)
class PurchaseResult:
    plans: tuple[PurchasePlan, ...]
    best: PurchasePlan
    baseline: Solution | None
    evaluations: int
    elapsed_s: float
    stop_reason: str  # "done", "time" or "cancelled"


def make_evaluator(problem: Problem, *, seed: int = 0, effort: str = "normal") -> Evaluator:
    """Deterministic, memoised estimate of the best loop an inventory allows (budgets in steps)."""
    expansions, iterations = EFFORTS[effort]
    cache: dict[tuple, Solution | None] = {}

    def evaluate(inventory: Mapping[str, int]) -> Solution | None:
        key = tuple(sorted((k, v) for k, v in inventory.items() if v > 0))
        if key in cache:
            return cache[key]
        view = dataclasses.replace(problem, inventory=dict(key))
        flat = dataclasses.replace(view, bridges=False)
        best: Solution | None = None
        starts: list[Solution] = list(generate_seeds(flat)[:1])
        if view.bridges:
            starts += bridge_seeds(view, limit=1)
        beam = beam_search(
            flat,
            seed=seed,
            config=BeamConfig(width=60, max_expansions=expansions, use_seeds=True),
        ).best
        if beam is not None:
            starts.append(beam)
        for start in starts:
            candidate = lns(
                view if _raised(start) else flat,
                seed=seed,
                config=LNSConfig(
                    max_iterations=iterations, bridge_share=0.4 if _raised(start) else 0.0
                ),
                initial=start,
            ).best
            if candidate is not None and (best is None or candidate.length_mm > best.length_mm):
                best = candidate
        cache[key] = best
        return best

    return evaluate


def _raised(solution: Solution) -> bool:
    return any(piece.level_in or piece.level_out for piece, _ in solution.items)


def _elevated(solution: Solution | None) -> int:
    if solution is None:
        return 0
    return sum(1 for piece, _ in solution.items if piece.level_in or piece.level_out)


def plan_purchases(
    problem: Problem,
    options: Sequence[PurchaseOption],
    budget: float,
    *,
    support_price: float = 0.0,
    evaluator: Evaluator | None = None,
    seed: int = 0,
    effort: str = "normal",
    time_budget_s: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[PurchasePlan], None] | None = None,
) -> PurchaseResult:
    """Plan purchases within ``budget``; only options with a positive price are considered."""
    if budget < 0:
        raise ValueError("budget must be >= 0")
    if support_price < 0:
        raise ValueError("support price must be >= 0")
    started = time.monotonic()
    evaluate = evaluator or make_evaluator(problem, seed=seed, effort=effort)
    offered = [o for o in options if o.price > 0]
    evaluations = 0
    stop_reason = "done"
    supports_owned = ELEVATED_KEY in problem.inventory  # a cap means supports are already owned

    def over_time() -> str | None:
        nonlocal stop_reason
        if should_stop is not None and should_stop():
            stop_reason = "cancelled"
        elif time_budget_s is not None and time.monotonic() - started >= time_budget_s:
            stop_reason = "time"
        return stop_reason if stop_reason != "done" else None

    def value(inventory: Mapping[str, int]) -> Solution | None:
        nonlocal evaluations
        evaluations += 1
        return evaluate(inventory)

    def support_cost(solution: Solution | None) -> float:
        return 0.0 if supports_owned else support_price * _elevated(solution)

    base_inventory = dict(problem.inventory)
    baseline = value(base_inventory)
    base_length = baseline.length_mm if baseline is not None else 0.0

    def run(strategy: str) -> PurchasePlan:
        inventory = dict(base_inventory)
        packs: Counter[str] = Counter()
        steps: list[PurchaseStep] = []
        spent, solution, length = 0.0, baseline, base_length

        def better(gain: float, price: float) -> float:
            return gain / price if strategy == "ratio" else gain

        while not over_time():
            best: tuple[float, tuple[PurchaseOption, ...], Solution] | None = None
            groups: list[tuple[PurchaseOption, ...]] = [(o,) for o in offered]
            for group_size in (1, 2):
                if group_size == 2:
                    if best is not None:
                        break
                    groups = [pair for pair in itertools.combinations_with_replacement(offered, 2)]
                for group in groups:
                    if over_time():
                        break
                    price = sum(o.price for o in group)
                    if spent + price > budget + 1e-9:
                        continue
                    trial = dict(inventory)
                    for option in group:
                        for slug, qty in option.items:
                            trial[slug] = trial.get(slug, 0) + qty
                    candidate = value(trial)
                    if candidate is None:
                        continue
                    if spent + price + support_cost(candidate) > budget + 1e-9:
                        continue
                    gain = candidate.length_mm - length
                    if gain <= EPSILON_MM:
                        continue
                    score = better(gain, price)
                    if best is None or score > best[0] + 1e-12:
                        best = (score, group, candidate)
            if best is None:
                break
            _, group, candidate = best
            price = sum(o.price for o in group)
            for option in group:
                packs[option.ref] += 1
                for slug, qty in option.items:
                    inventory[slug] = inventory.get(slug, 0) + qty
            spent += price
            steps.append(
                PurchaseStep(
                    tuple(o.ref for o in group),
                    tuple(o.name for o in group),
                    price,
                    spent,
                    candidate.length_mm,
                    candidate.length_mm - length,
                )
            )
            solution, length = candidate, candidate.length_mm
            if on_progress is not None:
                on_progress(
                    PurchasePlan(
                        strategy,
                        tuple(steps),
                        dict(packs),
                        dict(inventory),
                        solution,
                        spent,
                        support_cost(solution),
                        base_length,
                    )
                )
        return PurchasePlan(
            strategy, tuple(steps), dict(packs), inventory, solution,
            spent, support_cost(solution), base_length,
        )  # fmt: skip

    plans = (run("ratio"), run("gain"))
    best = max(plans, key=lambda p: (round(p.length_mm, 6), -p.total_cost))
    return PurchaseResult(
        plans, best, baseline, evaluations, time.monotonic() - started, stop_reason
    )


@dataclass(frozen=True)
class PurchaseSnapshot:
    status: str  # "idle", "running", "done", "cancelled", "failed"
    elapsed_s: float
    partial: PurchasePlan | None
    result: PurchaseResult | None
    error: str | None


@dataclass
class PurchaseJob:
    """A purchase plan computed in a background thread, with progress and cancellation."""

    problem: Problem
    options: Sequence[PurchaseOption]
    budget: float
    support_price: float = 0.0
    seed: int = 0
    effort: str = "normal"
    time_budget_s: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _started: float = 0.0
    _finished: float | None = None
    _partial: PurchasePlan | None = None
    _result: PurchaseResult | None = None
    _error: str | None = None
    _status: str = "idle"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("job already started")
        self._status, self._started = "running", time.monotonic()
        self._thread = threading.Thread(target=self._run, name="slotor-purchase", daemon=True)
        self._thread.start()

    def _progress(self, plan: PurchasePlan) -> None:
        with self._lock:
            if self._partial is None or plan.length_mm >= self._partial.length_mm:
                self._partial = plan

    def _run(self) -> None:
        try:
            result = plan_purchases(
                self.problem,
                self.options,
                self.budget,
                support_price=self.support_price,
                seed=self.seed,
                effort=self.effort,
                time_budget_s=self.time_budget_s,
                should_stop=self._stop.is_set,
                on_progress=self._progress,
            )
        except Exception as exc:  # reported, never swallowed
            with self._lock:
                self._error, self._status = f"{type(exc).__name__}: {exc}", "failed"
                self._finished = time.monotonic()
            return
        with self._lock:
            self._result = result
            self._status = "cancelled" if result.stop_reason == "cancelled" else "done"
            self._finished = time.monotonic()

    def cancel(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> PurchaseSnapshot:
        with self._lock:
            end = self._finished if self._finished is not None else time.monotonic()
            elapsed = end - self._started if self._started else 0.0
            return PurchaseSnapshot(self._status, elapsed, self._partial, self._result, self._error)
