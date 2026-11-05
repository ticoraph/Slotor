"""Large Neighborhood Search: destroy a window of the loop, repair it, keep only improvements.

The current solution is a closed loop, so any rotation of it is the same loop. A *window* of ``k``
consecutive pieces is removed; the remaining chain starts at the connector ``B`` where the window
ended and finishes at the connector ``A`` where it began. The repair is a short beam search that
extends that chain from ``A`` back to exactly ``B`` using the freed pieces and the unused stock.
Every candidate goes through the full validator before it can replace the current solution
(invariant 1), so an invalid state is never accepted.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from slotor.optimizers.beam import BeamConfig, Budget, HistoryPoint, run_beam
from slotor.optimizers.bridging import bridge_seeds
from slotor.optimizers.problem import Problem, Solution, make_solution
from slotor.optimizers.seeds import generate_seeds
from slotor.validation import IncrementalValidator

OPERATORS = ("window", "zone", "inefficient")


@dataclass(frozen=True)
class LNSConfig:
    time_budget_s: float | None = None
    max_iterations: int | None = None
    k_min: int = 2
    k_start: int = 6
    k_max: int = 16
    grow_after: int = 8  # consecutive failures before the window may grow by one piece
    repair_width: int = 40
    repair_expansions: int = 3000
    stagnation_limit: int = 80  # iterations without a new best before a restart
    kick_floor: float = 0.9  # a kick may return a loop down to this share of the best length
    weights: tuple[tuple[str, int], ...] = (("window", 4), ("zone", 3), ("inefficient", 2))
    use_seeds: bool = True
    # Adaptive selection: a Thompson-sampling bandit picks (operator, window size) from the
    # observed success rates of each arm of this run (no learning across instances).
    adaptive: bool = False
    # With bridges allowed, share of the iterations whose repair may use ramps and decks; the others
    # repair on the ground only, which keeps the flat search as sharp as without bridges.
    bridge_share: float = 0.4
    should_stop: Callable[[], bool] | None = None  # polled in the loop (cooperative cancellation)
    on_improve: Callable[[Solution], None] | None = None  # called with every new best solution

    def __post_init__(self) -> None:
        if not 1 <= self.k_min <= self.k_start <= self.k_max:
            raise ValueError("require 1 <= k_min <= k_start <= k_max")
        unknown = {name for name, _ in self.weights} - set(OPERATORS)
        if unknown:
            raise ValueError(f"unknown operators: {sorted(unknown)}")


K_BUCKETS = ((2, 4), (5, 8), (9, 12), (13, 16))


class _Bandit:
    """Beta-Bernoulli Thompson sampling over (operator, window-size bucket) arms."""

    def __init__(self, operators: list[str], k_max: int) -> None:
        self.arms = [
            (op, (lo, min(hi, k_max))) for op in operators for lo, hi in K_BUCKETS if lo <= k_max
        ]
        self.wins = {arm: 1.0 for arm in self.arms}
        self.losses = {arm: 1.0 for arm in self.arms}

    def choose(self, rng: random.Random) -> tuple[str, tuple[int, int]]:
        return max(self.arms, key=lambda a: rng.betavariate(self.wins[a], self.losses[a]))

    def update(self, arm: tuple[str, tuple[int, int]], success: bool) -> None:
        (self.wins if success else self.losses)[arm] += 1.0


@dataclass
class LNSStats:
    iterations: int = 0
    accepted: int = 0
    restarts: int = 0
    attempts: dict[str, int] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LNSResult:
    best: Solution | None
    history: tuple[HistoryPoint, ...]
    iterations: int
    elapsed_s: float
    stop_reason: str  # "iterations", "time", "cancelled" or "no_start"
    seed: int
    stats: LNSStats


def _elevated(solution: Solution) -> int:
    """Raised pieces and ramps: the supports a bridge needs, minimised at equal length."""
    return sum(1 for piece, _ in solution.items if piece.level_in or piece.level_out)


def _window_start(
    operator: str, current: Solution, k: int, rng: random.Random, problem: Problem
) -> int:
    n = current.n_pieces
    if operator == "window":
        return rng.randrange(n)
    placed = current.validation.track.pieces  # type: ignore[union-attr]
    if operator == "zone":
        left, bottom, right, top = problem.surface.bounds
        px, py = rng.uniform(left, right), rng.uniform(bottom, top)
        center = min(
            range(n),
            key=lambda i: math.hypot(
                placed[i].outline.centroid.x - px, placed[i].outline.centroid.y - py
            ),
        )
    else:  # "inefficient": short pieces contribute least length per piece
        weights = [1.0 / placed[i].geometry.length_mm for i in range(n)]
        center = rng.choices(range(n), weights=weights)[0]
    return (center - k // 2) % n


def _repair(
    problem: Problem,
    current: Solution,
    start_index: int,
    k: int,
    *,
    config: LNSConfig,
    rng: random.Random,
    budget: Budget,
    floor: float | None,
) -> Solution | None:
    """Destroy ``k`` pieces from ``start_index`` and look for a better closed completion.

    With ``floor`` set (a kick) the replacement only has to be different and at least that long.
    """
    n = current.n_pieces
    track = current.validation.track
    assert track is not None
    # A loop must start at ground level: grow the window until it ends on a level-0 connector.
    while k < n - 1 and track.pieces[(start_index + k) % n].entry.level != 0:
        k += 1
    if track.pieces[(start_index + k) % n].entry.level != 0:
        return None
    if not problem.bridges and track.pieces[start_index].entry.level != 0:
        return None  # a ground-only repair cannot start from a raised connector
    rotated_items = current.items[start_index:] + current.items[:start_index]
    rotated_placed = track.pieces[start_index:] + track.pieces[:start_index]
    target = rotated_placed[k].entry  # connector B: where the removed window ended
    remaining = rotated_items[k:]
    root = IncrementalValidator(
        problem.inventory, surface=problem.surface, start=target, tol=problem.tol
    )
    for piece, mirrored in remaining:
        if root.add(piece, mirrored):
            return None  # numerical edge: skip this neighbourhood rather than trust it
    current_length = current.length_mm
    original = tuple((p.slug, m) for p, m in rotated_items)
    found: Solution | None = None

    def accept(items, _length) -> float | None:
        nonlocal found
        solution = make_solution(problem, items, target)
        if solution is None:
            return None
        if floor is None:
            tied = abs(solution.length_mm - current_length) <= 1e-9
            better = solution.length_mm > current_length + 1e-9 or (
                tied and (solution.n_pieces, _elevated(solution)) < (n, _elevated(current))
            )
        else:
            slugs = tuple((p.slug, m) for p, m in items)
            better = solution.length_mm >= floor - 1e-9 and slugs != original
        if better and (found is None or solution.length_mm > found.length_mm + 1e-9):
            found = solution
            return solution.length_mm
        return None

    run_beam(
        problem,
        root,
        target,
        config=BeamConfig(width=config.repair_width),
        rng=random.Random(rng.getrandbits(32)),
        budget=budget.child(config.repair_expansions),
        incumbent=current_length if floor is None else floor,
        accept=accept,
        inner=None,
        strict=False,
    )
    return found


def lns(
    problem: Problem,
    *,
    seed: int = 0,
    config: LNSConfig | None = None,
    initial: Solution | None = None,
) -> LNSResult:
    """Improve ``initial`` (default: the longest seed); deterministic given ``max_iterations``."""
    config = LNSConfig() if config is None else config
    rng = random.Random(seed)
    budget = Budget(config.time_budget_s, None, config.should_stop)
    stats = LNSStats()
    seeds = generate_seeds(problem) if config.use_seeds else []
    if config.use_seeds and problem.bridges:
        seeds = sorted([*bridge_seeds(problem), *seeds], key=lambda s: (-s.length_mm, s.n_pieces))
    current = initial if initial is not None else (seeds[0] if seeds else None)
    if current is None:
        return LNSResult(None, (), 0, budget.elapsed(), "no_start", seed, stats)

    best = current
    history = [HistoryPoint(budget.elapsed(), 0, best.length_mm, best.n_pieces)]
    operators = [name for name, _ in config.weights]
    weights = [w for _, w in config.weights]
    k_cap = config.k_start
    bandit = _Bandit(operators, config.k_max) if config.adaptive else None
    flat_view = dataclasses.replace(problem, bridges=False)
    failures = 0
    stagnation = 0
    stop = "iterations"
    while True:
        if config.max_iterations is not None and stats.iterations >= config.max_iterations:
            break
        reason = budget.exhausted()
        if reason:
            stop = reason
            break
        stats.iterations += 1
        stagnation += 1
        n = current.n_pieces
        if n < 2:
            break
        if bandit is not None:
            arm = bandit.choose(rng)
            operator = arm[0]
            k = rng.randint(min(arm[1][0], n - 1), min(arm[1][1], n - 1))
        else:
            operator = rng.choices(operators, weights=weights)[0]
            k = rng.randint(min(config.k_min, n - 1), min(k_cap, n - 1))
        index = _window_start(operator, current, k, rng, problem)
        stats.attempts[operator] = stats.attempts.get(operator, 0) + 1
        use_bridges = problem.bridges and rng.random() < config.bridge_share
        candidate = _repair(
            problem if use_bridges else flat_view,
            current,
            index,
            k,
            config=config,
            rng=rng,
            budget=budget,
            floor=None,
        )
        if bandit is not None:
            bandit.update(arm, candidate is not None)
        if candidate is not None:
            stats.accepted += 1
            stats.successes[operator] = stats.successes.get(operator, 0) + 1
            current = candidate
            failures = 0
            k_cap = config.k_start
            if best.length_mm < current.length_mm - 1e-9:
                best = current
                stagnation = 0
                history.append(
                    HistoryPoint(budget.elapsed(), stats.iterations, best.length_mm, best.n_pieces)
                )
                if config.on_improve is not None:
                    config.on_improve(best)
        else:
            failures += 1
            if failures % config.grow_after == 0:
                k_cap = min(k_cap + 1, config.k_max)
        if stagnation >= config.stagnation_limit:
            stats.restarts += 1
            stagnation = 0
            failures = 0
            k_cap = config.k_start
            alternate = [s for s in seeds if s.slugs != current.slugs]
            if stats.restarts % 2 == 1 and alternate:
                current = alternate[(stats.restarts // 2) % len(alternate)]
            else:  # kick: a different loop of the best neighbourhood, allowed to be a bit shorter
                size = min(config.k_max, best.n_pieces - 1)
                start = rng.randrange(best.n_pieces)
                kicked = _repair(
                    problem,
                    best,
                    start,
                    size,
                    config=config,
                    rng=rng,
                    budget=budget,
                    floor=config.kick_floor * best.length_mm,
                )
                current = kicked if kicked is not None else best
    return LNSResult(best, tuple(history), stats.iterations, budget.elapsed(), stop, seed, stats)
