"""Inference with a trained MaskablePPO policy: rollouts and a prior for the Beam Search."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np

from slotor.catalog.model import Catalog
from slotor.envs.construction import ActionTable, ConstructionState
from slotor.optimizers.problem import Problem, Solution
from slotor.validation import IncrementalValidator


def state_from_validator(
    problem: Problem, table: ActionTable, validator: IncrementalValidator
) -> ConstructionState:
    """Rebuild a construction state from a chain (to evaluate the policy on a beam node)."""
    state = ConstructionState.initial(problem, table)
    for piece, mirrored in validator.items:
        index = table.entries.index((piece.slug, mirrored))
        state = state.children()[index]
    return state


class PolicyAgent:
    """A trained policy (CPU, one thread). The action table is checked against the catalog."""

    def __init__(self, model_path: str | Path, catalog: Catalog) -> None:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        self._torch = torch
        self.model = MaskablePPO.load(str(model_path), device="cpu")
        self.table = ActionTable.from_catalog(catalog)
        if self.model.action_space.n != len(self.table):
            raise ValueError("policy action space does not match the catalog's action table")

    def probabilities(self, state: ConstructionState) -> np.ndarray:
        """Policy distribution over the valid actions (zero elsewhere)."""
        mask = state.action_mask()
        if not mask.any():
            return np.zeros(len(self.table))
        torch = self._torch
        obs = torch.as_tensor(state.observation()[None, :])
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs, action_masks=mask[None, :])
            probs = dist.distribution.probs[0].numpy()
        return probs.astype(np.float64)

    def log_prior(self, problem: Problem):
        """A ``BeamConfig.prior`` callable: log-probability of each next ``(slug, mirrored)``."""

        def prior(validator: IncrementalValidator) -> dict[tuple[str, bool], float]:
            state = state_from_validator(problem, self.table, validator)
            probs = self.probabilities(state)
            return {
                self.table.entries[i]: float(np.log(max(p, 1e-9)))
                for i, p in enumerate(probs)
                if p > 0
            }

        return prior

    def rollout(
        self, problem: Problem, rng: random.Random, greedy: bool = False
    ) -> ConstructionState:
        state = ConstructionState.initial(problem, self.table)
        while True:
            probs = self.probabilities(state)
            if probs.sum() <= 0:
                return state
            if greedy:
                action = int(np.argmax(probs))
            else:
                action = rng.choices(range(len(probs)), weights=probs)[0]
            state = state.children()[action]
            if state.closed:
                return state

    def best_of_rollouts(
        self, problem: Problem, rng: random.Random, budget_s: float
    ) -> tuple[Solution | None, list[tuple[float, float]]]:
        """Sample loops until the time budget is spent; keep the longest validated one."""
        started = time.monotonic()
        best: Solution | None = None
        history: list[tuple[float, float]] = []
        first = True
        while first or time.monotonic() - started < budget_s:
            state = self.rollout(problem, rng, greedy=first)
            first = False
            if state.closed:
                solution = state.solution()
                if solution is not None and (best is None or solution.length_mm > best.length_mm):
                    best = solution
                    history.append((time.monotonic() - started, solution.length_mm))
        return best, history
