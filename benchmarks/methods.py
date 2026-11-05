"""The methods compared by the benchmark. Each takes a problem, a seed and a wall-clock budget.

All of them return a validated ``Solution`` or ``None``; the runner re-validates the exported
JSON with the full validator, so a method cannot inflate its own score (invariant 1).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from slotor.catalog.model import Catalog
from slotor.envs.construction import ActionTable, ConstructionState
from slotor.optimizers import BeamConfig, LNSConfig, Problem, Solution, beam_search, lns
from slotor.service import optimize

GREEDY_EPSILON = 0.15
BEAM_WIDTH = 200
PRIOR_WEIGHT_MM = 200.0  # mm of beam score per nat of learned log-probability


@dataclass
class MethodOutput:
    solution: Solution | None
    history: list[tuple[float, float]] = field(default_factory=list)  # (seconds, length_mm)


class Context:
    """Per-process resources: the catalog, the action table and the (lazily loaded) policy."""

    def __init__(self, catalog: Catalog, policy_path: str | Path | None) -> None:
        self.catalog = catalog
        self.table = ActionTable.from_catalog(catalog)
        self.policy_path = policy_path
        self._agent = None

    @property
    def agent(self):
        if self._agent is None:
            if self.policy_path is None:
                raise RuntimeError("this method needs a trained policy (--policy)")
            from slotor.rl.agent import PolicyAgent

            self._agent = PolicyAgent(self.policy_path, self.catalog)
        return self._agent


def _rollouts(ctx: Context, problem: Problem, budget_s: float, choose) -> MethodOutput:
    """Repeat constructive episodes until the budget is spent; keep the longest closed loop."""
    started = time.monotonic()
    best: Solution | None = None
    history: list[tuple[float, float]] = []
    episode = 0
    while episode == 0 or time.monotonic() - started < budget_s:
        state = ConstructionState.initial(problem, ctx.table)
        while True:
            children = state.children()
            if not children:
                break
            state = children[choose(state, children, episode)]
            if state.closed:
                break
        episode += 1
        if state.closed:
            solution = state.solution()
            if solution is not None and (best is None or solution.length_mm > best.length_mm):
                best = solution
                history.append((time.monotonic() - started, solution.length_mm))
    return MethodOutput(best, history)


def random_valid(ctx: Context, problem: Problem, seed: int, budget_s: float) -> MethodOutput:
    """Masked random rollouts: the "random valid" baseline."""
    rng = random.Random(seed)
    return _rollouts(ctx, problem, budget_s, lambda s, children, ep: rng.choice(sorted(children)))


def greedy(ctx: Context, problem: Problem, seed: int, budget_s: float) -> MethodOutput:
    """Longest valid piece first (epsilon-greedy after the first, deterministic, episode)."""
    rng = random.Random(seed)

    def choose(state: ConstructionState, children, episode: int) -> int:
        if episode > 0 and rng.random() < GREEDY_EPSILON:
            return rng.choice(sorted(children))
        # Gain = centerline length added by the child (curves have no ``length_mm``).
        gains = {i: child.length - state.length for i, child in children.items()}
        top = max(gains.values())
        return rng.choice(sorted(i for i, g in gains.items() if g >= top - 1e-9))

    return _rollouts(ctx, problem, budget_s, choose)


def _history(result_history) -> list[tuple[float, float]]:
    return [(h.elapsed_s, h.length_mm) for h in result_history]


def beam(ctx: Context, problem: Problem, seed: int, budget_s: float) -> MethodOutput:
    result = beam_search(
        problem, seed=seed, config=BeamConfig(width=BEAM_WIDTH, time_budget_s=budget_s)
    )
    return MethodOutput(result.best, _history(result.history))


def lns_method(ctx: Context, problem: Problem, seed: int, budget_s: float) -> MethodOutput:
    result = lns(problem, seed=seed, config=LNSConfig(time_budget_s=budget_s))
    return MethodOutput(result.best, _history(result.history))


def lns_bandit(ctx: Context, problem: Problem, seed: int, budget_s: float) -> MethodOutput:
    """LNS with Thompson-sampling selection of operator and window size (not formulation c)."""
    result = lns(problem, seed=seed, config=LNSConfig(time_budget_s=budget_s, adaptive=True))
    return MethodOutput(result.best, _history(result.history))


def pipeline(ctx: Context, problem: Problem, seed: int, budget_s: float) -> MethodOutput:
    seen: list[tuple[float, float]] = []
    started = time.monotonic()
    result = optimize(
        problem,
        seed=seed,
        time_budget_s=budget_s,
        top_k=1,
        on_progress=lambda s, stage: seen.append((time.monotonic() - started, s.length_mm)),
    )
    return MethodOutput(result.best, seen)


def rl_policy(ctx: Context, problem: Problem, seed: int, budget_s: float) -> MethodOutput:
    best, history = ctx.agent.best_of_rollouts(problem, random.Random(seed), budget_s)
    return MethodOutput(best, history)


def beam_prior(ctx: Context, problem: Problem, seed: int, budget_s: float) -> MethodOutput:
    result = beam_search(
        problem,
        seed=seed,
        config=BeamConfig(
            width=BEAM_WIDTH,
            time_budget_s=budget_s,
            prior=ctx.agent.log_prior(problem),
            prior_weight=PRIOR_WEIGHT_MM,
        ),
    )
    return MethodOutput(result.best, _history(result.history))


METHODS = {
    "random": random_valid,
    "greedy": greedy,
    "beam": beam,
    "lns": lns_method,
    "lns_bandit": lns_bandit,
    "pipeline": pipeline,
    "rl": rl_policy,
    "beam_prior": beam_prior,
}
NEEDS_POLICY = {"rl", "beam_prior"}
