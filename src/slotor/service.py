"""Application service: the one API the UI (and any future front end) calls (D2, D6).

It builds problems from user input, runs the optimization pipeline (seeds, Beam Search, LNS) in the
background, and returns validated solutions. Nothing here is specific to Gradio.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from slotor.catalog import Catalog, import_seed, load_catalog, parse_ta_export
from slotor.catalog.inventory import InventoryParseError
from slotor.catalog.model import ELEVATED_KEY
from slotor.db import apply_migrations, connect
from slotor.geometry import local_geometry
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
from slotor.validation import Surface

_BEAM_SHARE = 0.2  # share of the time budget given to the Beam Search stage
_BEAM_WIDTH = 150


def default_catalog() -> Catalog:
    """The packaged catalog, loaded into immutable objects (D18: no database in hot loops)."""
    connection = connect(":memory:")
    try:
        apply_migrations(connection)
        import_seed(connection)
        name = connection.execute("SELECT name FROM catalog_version").fetchone()["name"]
        return load_catalog(connection, name)
    finally:
        connection.close()


def optimizable_pieces(catalog: Catalog):
    """Pieces the optimizers can place: verified, with a geometry model."""
    probe = Problem(catalog, {p.slug: 1 for p in catalog.pieces}, Surface(1000.0, 1000.0))
    return probe.usable


def inventory_rows(catalog: Catalog, inventory: dict[str, int] | None = None) -> list[list[object]]:
    """Rows ``[slug, name, quantity]`` of the editable inventory table."""
    counts = inventory or {}
    return [[p.slug, p.name, int(counts.get(p.slug, 0))] for p in optimizable_pieces(catalog)]


def inventory_from_rows(catalog: Catalog, rows: Iterable[Sequence[object]]) -> dict[str, int]:
    """Inventory from table rows; blank quantities are zero, anything else must be a count."""
    inventory: dict[str, int] = {}
    for row in rows:
        slug = str(row[0]).strip()
        raw = row[-1]
        if slug == "" or slug == "nan":
            continue
        if slug not in catalog:
            raise ValueError(f"unknown piece {slug!r}")
        if raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == "":
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"quantity for {slug!r} is not a number: {raw!r}") from exc
        if number < 0 or number != int(number):
            raise ValueError(f"quantity for {slug!r} must be a whole number >= 0, got {raw!r}")
        if number:
            inventory[slug] = inventory.get(slug, 0) + int(number)
    return inventory


def import_ta_export(catalog: Catalog, text: str) -> dict[str, int]:
    """Parse a Track Architect text export; refuses lines it cannot match (never guesses)."""
    try:
        return parse_ta_export(text, catalog)
    except InventoryParseError as exc:
        raise ValueError(str(exc)) from exc


@dataclass(frozen=True)
class OptimizeRequest:
    surface_width_mm: float
    surface_height_mm: float
    margin_mm: float
    inventory: dict[str, int]
    time_budget_s: float = 10.0
    seed: int = 0
    top_k: int = 3
    lane_spacing_mm: float | None = None
    allow_bridges: bool = False
    max_elevated: int | None = None  # cap on raised pieces and ramps (None: unlimited)


def build_problem(catalog: Catalog, request: OptimizeRequest) -> Problem:
    if request.time_budget_s <= 0:
        raise ValueError("time budget must be positive")
    if not 1 <= request.top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")
    if request.lane_spacing_mm is not None and request.lane_spacing_mm <= 0:
        raise ValueError("lane spacing must be positive")
    inventory = {slug: n for slug, n in request.inventory.items() if n > 0}
    if not inventory:
        raise ValueError("the inventory is empty")
    try:
        surface = Surface(request.surface_width_mm, request.surface_height_mm, request.margin_mm)
    except ValueError as exc:
        raise ValueError(f"invalid surface: {exc}") from exc
    if request.max_elevated is not None:
        if request.max_elevated < 0:
            raise ValueError("max_elevated must be >= 0")
        inventory[ELEVATED_KEY] = request.max_elevated
    problem = Problem(catalog, inventory, surface, bridges=request.allow_bridges)
    if not problem.usable:
        raise ValueError("no piece of the inventory can be used (none has a geometry model)")
    return problem


@dataclass(frozen=True)
class OptimizationResult:
    top: tuple[Solution, ...]  # distinct loops, longest first
    stages: tuple[tuple[str, float], ...]  # (stage, best length after it)
    elapsed_s: float
    stop_reason: str  # "done" or "cancelled"
    message: str

    @property
    def best(self) -> Solution | None:
        return self.top[0] if self.top else None


def _has_bridge(solution: Solution) -> bool:
    return any(piece.level_in or piece.level_out for piece, _ in solution.items)


def _key(solution: Solution) -> tuple:
    """Rotation-independent identity of a loop: length, size and parts list."""
    used = tuple(sorted(Counter(piece.slug for piece, _ in solution.items).items()))
    return (round(solution.length_mm, 3), solution.n_pieces, used)


def optimize(
    problem: Problem,
    *,
    seed: int = 0,
    time_budget_s: float = 10.0,
    top_k: int = 3,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[Solution, str], None] | None = None,
) -> OptimizationResult:
    """Seeds, then a short Beam Search, then ``top_k`` LNS runs sharing the remaining time."""
    started = time.monotonic()
    deadline = started + time_budget_s
    candidates: list[Solution] = []
    stages: list[tuple[str, float]] = []
    overall: Solution | None = None

    def offer(solution: Solution, stage: str) -> None:
        nonlocal overall
        if overall is None or solution.length_mm > overall.length_mm + 1e-9:
            overall = solution
            if on_progress is not None:
                on_progress(solution, stage)

    def cancelled() -> bool:
        return should_stop is not None and should_stop()

    # No loop can be longer than every usable piece laid end to end: stop when it is reached.
    upper_mm = sum(
        local_geometry(piece, False, problem.tol).length_mm * count
        for piece, count in ((p, problem.stock[p.slug]) for p in problem.flat_usable)
    )

    def bound_reached() -> bool:
        return overall is not None and overall.length_mm >= upper_mm - 1e-6

    def stop_now() -> bool:
        return cancelled() or bound_reached()

    seeds = generate_seeds(problem)
    for solution in seeds[:top_k]:
        candidates.append(solution)
    if seeds:
        offer(seeds[0], "seed")
        stages.append(("seed", seeds[0].length_mm))

    beam_budget = max(0.05, time_budget_s * _BEAM_SHARE)
    # The Beam Search runs on the ground only (bridges multiply its branching); the LNS then
    # explores ramps and decks starting from its result.
    beam = beam_search(
        dataclasses.replace(problem, bridges=False),
        seed=seed,
        config=BeamConfig(
            width=_BEAM_WIDTH,
            time_budget_s=beam_budget,
            should_stop=stop_now,
            on_improve=lambda s: offer(s, "beam"),
        ),
    )
    if beam.best is not None:
        candidates.append(beam.best)
        stages.append(("beam", beam.best.length_mm))

    bridged = bridge_seeds(problem, limit=max(1, top_k // 2)) if problem.bridges else []
    for solution in bridged:
        candidates.append(solution)
        offer(solution, "bridge")
    if bridged:
        stages.append(("bridge", bridged[0].length_mm))
    # LNS runs start from distinct loops: bridged seeds first, then the beam result, then seeds.
    starts: list[Solution] = list(bridged)
    starts += [s for s in ([beam.best] if beam.best else [])]
    starts += [s for s in seeds[1:top_k] if s is not None]
    if not starts and seeds:
        starts = [seeds[0]]
    runs = max(1, min(top_k, len(starts)))
    for index in range(runs):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or stop_now() or not starts:
            break
        share = remaining / (runs - index)
        start = starts[index]
        # A start that holds a bridge keeps exploring bridges. A flat start is improved on the
        # ground only when bridged seeds exist or when it is the first run, so allowing bridges
        # does not take search effort away from the flat result; other flat starts try bridges.
        ground_only = not _has_bridge(start) and (bool(bridged) or index == 0)
        result = lns(
            dataclasses.replace(problem, bridges=False) if ground_only else problem,
            seed=seed + index,
            initial=start,
            config=LNSConfig(
                time_budget_s=share,
                bridge_share=0.4 if _has_bridge(start) else 0.15,
                should_stop=stop_now,
                on_improve=lambda s: offer(s, "lns"),
            ),
        )
        if result.best is not None:
            candidates.append(result.best)
        if result.stop_reason == "cancelled" and stop_now():
            break
    if overall is not None:
        stages.append(("lns", overall.length_mm))

    distinct: dict[tuple, Solution] = {}
    for solution in candidates:
        distinct.setdefault(_key(solution), solution)
    top = tuple(sorted(distinct.values(), key=lambda s: (-s.length_mm, s.n_pieces))[:top_k])
    was_cancelled = cancelled()
    if top:
        message = (
            f"{len(top)} circuit(s) valide(s), le plus long fait {top[0].length_mm / 1000:.3f} m."
        )
        if bound_reached():
            message += " Tout l'inventaire est utilisé : impossible de faire plus long."
    else:
        message = "Aucun circuit fermé valide trouvé avec cet inventaire et cette surface."
    return OptimizationResult(
        top,
        tuple(stages),
        time.monotonic() - started,
        "cancelled" if was_cancelled else "done",
        message,
    )


@dataclass(frozen=True)
class JobSnapshot:
    status: str  # "idle", "running", "done", "cancelled" or "failed"
    elapsed_s: float
    stage: str
    best: Solution | None
    result: OptimizationResult | None
    error: str | None


@dataclass
class OptimizationJob:
    """An optimization running in a background thread, with progressive best and cancellation."""

    problem: Problem
    seed: int = 0
    time_budget_s: float = 10.0
    top_k: int = 3
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _started: float = 0.0
    _finished: float | None = None
    _stage: str = ""
    _best: Solution | None = None
    _result: OptimizationResult | None = None
    _error: str | None = None
    _status: str = "idle"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("job already started")
        self._status = "running"
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="slotor-optimize", daemon=True)
        self._thread.start()

    def _progress(self, solution: Solution, stage: str) -> None:
        with self._lock:
            self._best, self._stage = solution, stage

    def _run(self) -> None:
        try:
            result = optimize(
                self.problem,
                seed=self.seed,
                time_budget_s=self.time_budget_s,
                top_k=self.top_k,
                should_stop=self._stop.is_set,
                on_progress=self._progress,
            )
        except Exception as exc:  # reported to the UI, never swallowed silently
            with self._lock:
                self._error, self._status = f"{type(exc).__name__}: {exc}", "failed"
                self._finished = time.monotonic()
            return
        with self._lock:
            self._result = result
            self._best = result.best or self._best
            self._status = "cancelled" if result.stop_reason == "cancelled" else "done"
            self._finished = time.monotonic()

    def cancel(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            end = self._finished if self._finished is not None else time.monotonic()
            elapsed = end - self._started if self._started else 0.0
            return JobSnapshot(
                self._status, elapsed, self._stage, self._best, self._result, self._error
            )


def import_summary(catalog: Catalog, inventory: dict[str, int]) -> str:
    """Markdown note after an import: what was read and what the optimizers will ignore."""
    usable = {p.slug for p in optimizable_pieces(catalog)}
    kept = sum(n for slug, n in inventory.items() if slug in usable)
    ignored = {slug: n for slug, n in inventory.items() if slug not in usable}
    text = f"Import : {kept} pièces utilisables sur {sum(inventory.values())}."
    if ignored:
        text += " Ignorées (pas encore modélisées) : " + ", ".join(
            f"{slug} × {n}" for slug, n in sorted(ignored.items())
        )
        text += "."
    return text


@dataclass(frozen=True)
class Presentation:
    """Everything a front end needs to display a job, as plain strings and rows."""

    status: str
    summary: str
    drawings_html: str
    usage: list[list[object]]
    bom: str
    json_text: str | None
    svg_text: str | None


def present(problem: Problem, snapshot: JobSnapshot, lane_spacing_mm: float | None) -> Presentation:
    from slotor import report as rp

    if snapshot.status == "failed":
        return Presentation(f"Échec : {snapshot.error}", "", "", [], "", None, None)
    solutions: tuple[Solution, ...] = ()
    if snapshot.result is not None:
        solutions = snapshot.result.top
    elif snapshot.best is not None:
        solutions = (snapshot.best,)
    best = solutions[0] if solutions else None
    labels = {
        "idle": "En attente",
        "running": "Optimisation en cours",
        "done": "Terminé",
        "cancelled": "Arrêté",
    }
    status = f"{labels.get(snapshot.status, snapshot.status)} — {snapshot.elapsed_s:.0f} s"
    if best is not None:
        status += f" — meilleur : {best.length_mm / 1000:.3f} m"
        if snapshot.status == "running" and snapshot.stage:
            status += f" ({snapshot.stage})"
    if snapshot.result is not None:
        status += f". {snapshot.result.message}"
    if best is None:
        return Presentation(status, "", "", [], "", None, None)
    figures = rp.build_report(best, problem, lane_spacing_mm)
    blocks = []
    for rank, solution in enumerate(solutions, start=1):
        caption = (
            f"<p><b>#{rank}</b> — {solution.length_mm / 1000:.3f} m, {solution.n_pieces} pièces</p>"
        )
        # Fit the column in the page: the SVG keeps its aspect ratio through its viewBox.
        drawing = rp.solution_to_svg(solution, problem).replace(
            "<svg ", '<svg style="max-width:100%;height:auto" ', 1
        )
        blocks.append(f"<div style='margin-bottom:16px'>{caption}{drawing}</div>")
    return Presentation(
        status,
        rp.format_summary(figures),
        "".join(blocks),
        rp.usage_rows(figures),
        rp.bill_of_materials(figures),
        rp.solution_to_json(best, problem, lane_spacing_mm),
        rp.solution_to_svg(best, problem),
    )


def write_exports(presentation: Presentation, directory) -> tuple[str | None, str | None]:
    """Write the JSON and SVG of the best solution; returns their paths (or ``None``)."""
    from pathlib import Path

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[str | None] = []
    for name, content in (
        ("slotor-circuit.json", presentation.json_text),
        ("slotor-circuit.svg", presentation.svg_text),
    ):
        if content is None:
            paths.append(None)
            continue
        target = folder / name
        target.write_text(content, encoding="utf-8")
        paths.append(str(target))
    return paths[0], paths[1]


# --- « Quels rails acheter ? » (phase 8) ---------------------------------------------------------


@dataclass(frozen=True)
class PurchaseRequest:
    surface_width_mm: float
    surface_height_mm: float
    margin_mm: float
    inventory: dict[str, int]
    prices: dict[str, float]  # official pack reference -> price; absent or 0 = not offered
    budget: float
    support_price: float = 0.0
    allow_bridges: bool = False
    max_elevated: int | None = None
    effort: str = "normal"
    seed: int = 0
    time_budget_s: float = 300.0


def purchase_price_rows(
    catalog: Catalog, prices: dict[str, float] | None = None
) -> list[list[object]]:
    """Rows ``[ref, name, content, price]`` of the editable price table."""
    from slotor.purchase import purchase_options

    rows = []
    for option in purchase_options(catalog, prices):
        content = ", ".join(f"{qty} × {catalog.get(slug).name}" for slug, qty in option.items)
        rows.append([option.ref, option.name, content, option.price or None])
    return rows


def prices_from_rows(rows: Iterable[Sequence[object]]) -> dict[str, float]:
    """Prices from table rows; blank means not offered, anything else must be a number >= 0."""
    prices: dict[str, float] = {}
    for row in rows:
        ref, raw = str(row[0]).strip(), row[-1]
        if ref in ("", "nan"):
            continue
        if raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"price for {ref} is not a number: {raw!r}") from exc
        if value < 0:
            raise ValueError(f"price for {ref} must be >= 0")
        if value > 0:
            prices[ref] = value
    return prices


def build_purchase(catalog: Catalog, request: PurchaseRequest):
    """The problem and the priced options of a purchase request (inventory may be empty)."""
    from slotor.purchase import EFFORTS, purchase_options

    if request.effort not in EFFORTS:
        raise ValueError(f"effort must be one of {sorted(EFFORTS)}")
    if request.budget <= 0:
        raise ValueError("the budget must be positive")
    options = purchase_options(catalog, request.prices)
    if not any(o.price > 0 for o in options):
        raise ValueError("no pack has a price: enter at least one price")
    unknown = set(request.prices) - {o.ref for o in options}
    if unknown:
        raise ValueError(f"packs that cannot be planned: {sorted(unknown)}")
    inventory = {slug: n for slug, n in request.inventory.items() if n > 0}
    if request.max_elevated is not None:
        if request.max_elevated < 0:
            raise ValueError("max_elevated must be >= 0")
        inventory[ELEVATED_KEY] = request.max_elevated
    try:
        surface = Surface(request.surface_width_mm, request.surface_height_mm, request.margin_mm)
    except ValueError as exc:
        raise ValueError(f"invalid surface: {exc}") from exc
    problem = Problem(catalog, inventory, surface, bridges=request.allow_bridges)
    return problem, options


PURCHASE_HEADERS = ["Étape", "Achat", "Prix", "Cumul", "Longueur (m)", "Gain (m)"]


@dataclass(frozen=True)
class PurchasePresentation:
    status: str
    summary: str
    steps: list[list[object]]
    shopping_list: str
    drawings_html: str
    bom: str


def present_purchase(problem: Problem, snapshot, request: PurchaseRequest) -> PurchasePresentation:
    from slotor import report as rp

    if snapshot.status == "failed":
        return PurchasePresentation(f"Échec : {snapshot.error}", "", [], "", "", "")
    labels = {
        "idle": "En attente",
        "running": "Calcul en cours",
        "done": "Terminé",
        "cancelled": "Arrêté",
    }
    status = f"{labels.get(snapshot.status, snapshot.status)} — {snapshot.elapsed_s:.0f} s"
    plan = snapshot.result.best if snapshot.result is not None else snapshot.partial
    if plan is None:
        return PurchasePresentation(status, "", [], "", "", "")

    names = {}
    for step in plan.steps:
        names.update(zip(step.refs, step.names, strict=True))
    rows = [
        [
            index,
            " + ".join(f"{ref} ({name})" for ref, name in zip(step.refs, step.names, strict=True)),
            step.price,
            step.cumulative_price,
            round(step.length_mm / 1000, 3),
            round(step.gain_mm / 1000, 3),
        ]
        for index, step in enumerate(plan.steps, start=1)
    ]
    prices = request.prices
    lines = [
        f"{count} × {ref} — {names.get(ref, ref)} — {prices[ref]:g} chacun, "
        f"soit {count * prices[ref]:g}"
        for ref, count in sorted(plan.packs.items())
    ]
    if plan.support_cost:
        lines.append(f"Supports : {plan.support_cost:g}")
    lines.append(f"Total : {plan.total_cost:g} (budget {request.budget:g})")
    base_m, new_m = plan.baseline_length_mm / 1000, plan.length_mm / 1000
    if plan.steps:
        percent = 100 * plan.gain_mm / max(plan.baseline_length_mm, 1)
        summary = (
            f"**Avec {plan.total_cost:g} dépensés, la ligne médiane passe de {base_m:.3f} m à "
            f"{new_m:.3f} m** (+{plan.gain_mm / 1000:.3f} m, +{percent:.0f} %)."
        )
    else:
        summary = (
            f"**Aucun achat ne rallonge le circuit** dans ce budget et cette surface "
            f"(longueur actuelle : {base_m:.3f} m)."
        )
    summary += (
        "\n\nLes prix sont ceux que vous avez saisis (le catalogue n'en contient pas) ; chaque "
        "longueur est celle d'un circuit validé. L'estimation est heuristique : un autre achat "
        "pourrait faire mieux."
    )
    drawings = ""
    after = plan.solution
    before = snapshot.result.baseline if snapshot.result is not None else None
    after_problem = dataclasses.replace(problem, inventory=plan.inventory)
    for title, solution, view in (
        ("Avant achat", before, problem),
        ("Après achat", after, after_problem),
    ):
        if solution is not None:
            svg = rp.solution_to_svg(solution, view).replace(
                "<svg ", '<svg style="max-width:100%;height:auto" ', 1
            )
            caption = f"<p><b>{title}</b> — {solution.length_mm / 1000:.3f} m</p>"
            drawings += f"<div style='margin-bottom:16px'>{caption}{svg}</div>"
    bom = ""
    if after is not None:
        bom = rp.bill_of_materials(rp.build_report(after, after_problem))
    return PurchasePresentation(status, summary, rows, "\n".join(lines), drawings, bom)


def create_purchase_job(problem: Problem, options, request: PurchaseRequest):
    """A purchase planning job for ``request`` (not started)."""
    from slotor.purchase import PurchaseJob

    return PurchaseJob(
        problem,
        options,
        request.budget,
        request.support_price,
        request.seed,
        request.effort,
        request.time_budget_s,
    )
