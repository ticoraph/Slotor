"""Constructive Beam Search with admissible pruning, bucketed diversity and an anytime result.

The search runs in free space from the pose ``(0, 0, 0)``: translation is free, so the surface
only bounds the *extent* of the layout (as is, or turned a quarter turn). Every candidate piece is
checked by the incremental validator; every reported solution is re-validated by ``finalize``
(full validator, surface and closure included). Results are deterministic for a given ``seed`` and
a ``max_expansions`` limit; a wall-clock budget makes them depend on the machine's speed.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from slotor.catalog.model import PieceType
from slotor.geometry import Pose, ToleranceConfig, local_geometry
from slotor.geometry.pose import wrap_angle
from slotor.optimizers.problem import Problem, Solution, finalize, fits
from slotor.optimizers.seeds import generate_seeds
from slotor.validation import IncrementalValidator

_ROOT = Pose(0.0, 0.0, 0.0, 0)
_CELL_MM = 200.0  # grain of the occupancy fingerprint used for state de-duplication
_QUANT_MM = 2.0
_QUANT_DEG = 0.5


@dataclass(frozen=True)
class BeamConfig:
    width: int = 200
    closure_weight: float = 0.5  # how strongly a state far from its start is demoted when ranking
    bucket_cell_mm: float = 500.0
    heading_bucket_deg: float = 60.0
    time_budget_s: float | None = None
    max_expansions: int | None = None
    use_seeds: bool = True
    should_stop: Callable[[], bool] | None = None  # polled in the loop (cooperative cancellation)
    on_improve: Callable[[Solution], None] | None = None  # called with every new best solution
    # Learned prior (formulation b): maps a chain to the log-probability of each next
    # ``(slug, mirrored)``; it only re-ranks children (``prior_weight`` * log-prob), never prunes.
    prior: Callable[[IncrementalValidator], dict[tuple[str, bool], float]] | None = None
    prior_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be at least 1")


@dataclass(frozen=True, slots=True)
class HistoryPoint:
    elapsed_s: float
    expansions: int
    length_mm: float
    n_pieces: int


@dataclass(frozen=True)
class SearchResult:
    best: Solution | None
    history: tuple[HistoryPoint, ...]
    expansions: int
    elapsed_s: float
    stop_reason: str  # "exhausted", "time", "expansions" or "cancelled"
    seed: int


def can_still_close(
    pose: Pose,
    start: Pose,
    remaining_length_mm: float,
    remaining_turn_deg: float,
    tol: ToleranceConfig,
    min_width_mm: float,
) -> bool:
    """Admissible closure bound: never rejects a state that can still be completed.

    Position: no piece moves its end further than its centerline length, so the distance to the
    start cannot exceed the remaining rail length (plus the closure play). Heading: the heading
    change still needed cannot exceed the remaining curve angle (plus the angular play).
    """
    distance = math.hypot(pose.x - start.x, pose.y - start.y)
    if distance - tol.closure_pos_mm > remaining_length_mm:
        return False
    needed = abs(math.degrees(wrap_angle(start.theta - pose.theta)))
    slack = math.degrees(tol.closure_angle_edge_mm / min_width_mm)
    return needed <= remaining_turn_deg + slack + 1e-9


@dataclass
class _Node:
    validator: IncrementalValidator
    length: float
    remaining_length: float
    remaining_turn: float
    left_count: int
    extent: tuple[float, float, float, float] | None
    cells: frozenset[tuple[int, int]]
    tie: float
    score: float = 0.0
    key: tuple = field(default=(), repr=False)


def _merge(
    extent: tuple[float, float, float, float] | None, box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    if extent is None:
        return box
    return (
        min(extent[0], box[0]),
        min(extent[1], box[1]),
        max(extent[2], box[2]),
        max(extent[3], box[3]),
    )


class Budget:
    """Wall-clock deadline and/or expansion limit shared by a search and its sub-searches."""

    def __init__(
        self,
        time_budget_s: float | None,
        max_expansions: int | None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.should_stop = should_stop
        self.started = time.monotonic()
        self.deadline = None if time_budget_s is None else self.started + time_budget_s
        self.max_expansions = max_expansions
        self.expansions = 0

    def exhausted(self) -> str | None:
        if self.should_stop is not None and self.should_stop():
            return "cancelled"
        if self.max_expansions is not None and self.expansions >= self.max_expansions:
            return "expansions"
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return "time"
        return None

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def child(self, max_expansions: int | None) -> Budget:
        """A sub-budget with its own expansion limit and the same deadline."""
        sub = Budget(None, max_expansions, self.should_stop)
        sub.deadline = self.deadline
        return sub


def run_beam(
    problem: Problem,
    root: IncrementalValidator,
    target: Pose,
    *,
    config: BeamConfig,
    rng: random.Random,
    budget: Budget,
    incumbent: float,
    accept: Callable[[tuple[tuple[PieceType, bool], ...], float], float | None],
    inner: tuple[float, float] | None,
    strict: bool = True,
) -> str:
    """Extend ``root`` (a chain that may already hold pieces) until it closes on ``target``.

    ``accept(items, length)`` is called for every closed, valid candidate at least as long as the
    incumbent; it returns the new incumbent length when it keeps the candidate, else ``None``.
    ``inner`` bounds the free-space extent (beam from scratch); ``None`` when the validator's own
    surface is absolute (repair). ``strict`` prunes branches that can only tie the incumbent.
    Returns the stop reason: ``exhausted``, ``time`` or ``expansions``.
    """
    tol = problem.tol
    pieces: list[PieceType] = list(problem.usable)
    stock = problem.stock
    geometry = {p.slug: local_geometry(p, False, tol) for p in pieces}
    piece_length = {slug: g.length_mm for slug, g in geometry.items()}
    piece_turn = {p.slug: (p.turn_angle_deg or 0.0) for p in pieces}
    min_width = min(g.width_mm for g in geometry.values())
    options = {p.slug: ((False,) if p.category == "straight" else (False, True)) for p in pieces}
    used0 = root.used
    left = {s: n - used0.get(s, 0) for s, n in stock.items()}
    remaining_length = sum(piece_length[s] * max(n, 0) for s, n in left.items())
    remaining_turn = sum(piece_turn[s] * max(n, 0) for s, n in left.items())
    prefix_length = sum(local_geometry(p, m, tol).length_mm for p, m in root.items)
    left_count0 = sum(max(n, 0) for n in left.values())
    extent = None
    for placed in root.placed:
        extent = _merge(extent, placed.outline.bounds)
    margin = 1e-9 if strict else -1e-9
    frontier = [
        _Node(
            root,
            prefix_length,
            remaining_length,
            remaining_turn,
            left_count0,
            extent,
            frozenset(),
            rng.random(),
        )
    ]
    stop = "exhausted"
    while frontier:
        children: dict[tuple, _Node] = {}
        for node in frontier:
            reason = budget.exhausted()
            if reason:
                stop = reason
                break
            order = list(pieces)
            rng.shuffle(order)
            log_prior = config.prior(node.validator) if config.prior is not None else {}
            for piece in order:
                if node.validator.used.get(piece.inventory_key, 0) >= stock[piece.inventory_key]:
                    continue
                for mirrored in options[piece.slug]:
                    budget.expansions += 1
                    child_validator = node.validator.fork()
                    if child_validator.add(piece, mirrored):
                        continue
                    placed = child_validator.placed[-1]
                    extent = _merge(node.extent, placed.outline.bounds)
                    if inner is not None and not fits(
                        (extent[2] - extent[0], extent[3] - extent[1]), inner
                    ):
                        continue
                    length = node.length + piece_length[piece.slug]
                    left_length = node.remaining_length - piece_length[piece.slug]
                    left_turn = node.remaining_turn - piece_turn[piece.slug]
                    pose = child_validator.end_pose
                    if _is_closed_candidate(pose, target, tol):
                        if length >= incumbent - 1e-9 and child_validator.result().valid:
                            kept = accept(child_validator.items, length)
                            if kept is not None:
                                incumbent = kept
                        continue  # a closed loop is never extended
                    if length + left_length <= incumbent + margin:
                        continue  # cannot beat the incumbent even with every remaining piece
                    if not can_still_close(pose, target, left_length, left_turn, tol, min_width):
                        continue
                    left_count = node.left_count - 1
                    if abs(pose.level - target.level) > left_count:
                        continue  # not enough pieces left to climb or descend back to the target
                    cx, cy = placed.outline.centroid.x, placed.outline.centroid.y
                    cells = node.cells | {(int(cx // _CELL_MM), int(cy // _CELL_MM))}
                    heading = math.degrees(pose.theta) % 360.0
                    detour = math.hypot(pose.x - target.x, pose.y - target.y) + 297.0 * abs(
                        wrap_angle(pose.theta - target.theta)
                    )
                    child = _Node(
                        child_validator,
                        length,
                        left_length,
                        left_turn,
                        left_count,
                        extent,
                        cells,
                        rng.random(),
                        score=length
                        - config.closure_weight * detour
                        + config.prior_weight * log_prior.get((piece.slug, mirrored), 0.0),
                    )
                    child.key = (
                        pose.level,
                        round(pose.x / _QUANT_MM),
                        round(pose.y / _QUANT_MM),
                        round(heading / _QUANT_DEG) % round(360.0 / _QUANT_DEG),
                        tuple(sorted(child_validator.used.items())),
                        cells,
                    )
                    known = children.get(child.key)
                    if known is None or (child.score, child.tie) > (known.score, known.tie):
                        children[child.key] = child
        if stop != "exhausted":
            break
        frontier = _select(list(children.values()), config)
        reason = budget.exhausted()
        if reason:
            stop = reason
            break
    return stop


def beam_search(
    problem: Problem, *, seed: int = 0, config: BeamConfig | None = None
) -> SearchResult:
    config = BeamConfig() if config is None else config
    rng = random.Random(seed)
    budget = Budget(config.time_budget_s, config.max_expansions, config.should_stop)
    history: list[HistoryPoint] = []
    best: Solution | None = None

    def record(solution: Solution) -> None:
        nonlocal best
        best = solution
        history.append(
            HistoryPoint(budget.elapsed(), budget.expansions, solution.length_mm, solution.n_pieces)
        )
        if config.on_improve is not None:
            config.on_improve(solution)

    if config.use_seeds:
        seeds = generate_seeds(problem, limit=1)
        if seeds:
            record(seeds[0])

    def accept(items, length) -> float | None:
        solution = finalize(problem, items)
        if solution is not None and (best is None or solution.length_mm > best.length_mm + 1e-9):
            record(solution)
            return solution.length_mm
        return None

    stop = "exhausted"
    if problem.usable:
        root = IncrementalValidator(problem.inventory, start=_ROOT, tol=problem.tol)
        stop = run_beam(
            problem,
            root,
            _ROOT,
            config=config,
            rng=rng,
            budget=budget,
            incumbent=best.length_mm if best is not None else -1.0,
            accept=accept,
            inner=problem.inner_size_mm,
        )
    return SearchResult(best, tuple(history), budget.expansions, budget.elapsed(), stop, seed)


def _is_closed_candidate(pose: Pose, target: Pose, tol: ToleranceConfig) -> bool:
    """Cheap pre-filter before the validator's own closure verdict (loose on purpose)."""
    return (
        pose.level == target.level
        and math.hypot(pose.x - target.x, pose.y - target.y) <= 2 * tol.closure_pos_mm
        and abs(wrap_angle(pose.theta - target.theta)) <= 1e-2
    )


def _select(nodes: list[_Node], config: BeamConfig) -> list[_Node]:
    """Best ``width`` nodes, round-robin over spatial/heading buckets for diversity."""
    if len(nodes) <= config.width:
        return nodes
    buckets: dict[tuple, list[_Node]] = defaultdict(list)
    for node in nodes:
        pose = node.validator.end_pose
        key = (
            int(pose.x // config.bucket_cell_mm),
            int(pose.y // config.bucket_cell_mm),
            int((math.degrees(pose.theta) % 360.0) // config.heading_bucket_deg),
        )
        buckets[key].append(node)
    ranked = [
        sorted(group, key=lambda n: (n.score, n.tie), reverse=True)
        for _, group in sorted(buckets.items(), key=lambda kv: kv[0])
    ]
    ranked.sort(key=lambda group: (group[0].score, group[0].tie), reverse=True)
    selected: list[_Node] = []
    depth = 0
    while len(selected) < config.width:
        added = False
        for group in ranked:
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == config.width:
                    break
        if not added:
            break
        depth += 1
    return selected
