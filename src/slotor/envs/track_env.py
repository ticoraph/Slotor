"""Gymnasium environment: build a closed loop piece by piece (formulation a, D11).

* Actions: one index per ``(piece, mirrored)`` (fixed table); the mask comes from the incremental
  validator plus the admissible closure bound, so an invalid layout can never be built.
* Reward: terminal ``length / total rail length`` when the loop closes (the validated length of
  the full validator), plus the potential-based shaping ``gamma * phi(s') - phi(s)`` (policy
  invariant, see ``_phi``). No reward for an open or invalid layout.
* Dead end: when no action is valid the episode ends with a moderate penalty; the agent never
  sees an all-false mask. A masked-out action is never applied: it ends the episode (penalty).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from slotor.catalog.model import Catalog
from slotor.envs.construction import OBS_SIZE, ActionTable, ConstructionState
from slotor.envs.instances import InstanceSampler
from slotor.geometry import local_geometry
from slotor.optimizers.problem import Problem, Solution
from slotor.optimizers.seeds import stadium_layouts

DEADLOCK_PENALTY = -0.2
INVALID_ACTION_PENALTY = -1.0


class TrackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        catalog: Catalog,
        *,
        problems: list[Problem] | None = None,
        sampler: InstanceSampler | None = None,
        gamma: float = 1.0,
        shaping: float = 0.5,
        length_shaping: float = 0.0,
        reward_reference: str = "stock",
        partial_start_prob: float = 0.0,
        max_removed: int = 8,
    ) -> None:
        super().__init__()
        if (problems is None) == (sampler is None):
            raise ValueError("give exactly one of problems (frozen suite) or sampler")
        self.catalog = catalog
        self.table = ActionTable.from_catalog(catalog)
        self.problems = problems
        self.sampler = sampler
        self.gamma = gamma
        self.shaping = shaping
        if reward_reference not in {"stock", "seed"}:
            raise ValueError("reward_reference must be 'stock' or 'seed'")
        self.length_shaping = length_shaping
        self.reward_reference = reward_reference
        self._reference = 1.0
        self.partial_start_prob = partial_start_prob
        self.max_removed = max_removed
        self.action_space = spaces.Discrete(len(self.table))
        self.observation_space = spaces.Box(-4.0, 4.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.state: ConstructionState | None = None
        self.last_solution: Solution | None = None
        self._cursor = 0

    # -- Gymnasium API ---------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if "problem" in options:
            problem = options["problem"]
        elif self.problems is not None:
            problem = self.problems[self._cursor % len(self.problems)]
            self._cursor += 1
        else:
            problem = self.sampler.sample(self.np_random)  # type: ignore[union-attr]
        self.last_solution = None
        state = ConstructionState.initial(problem, self.table)
        self._reference = self._reward_reference(problem, state)
        if self.partial_start_prob > 0 and self.np_random.random() < self.partial_start_prob:
            state = self._partial_start(problem) or state
        self.state = state
        return self._observation(), self._info()

    def step(self, action: int):
        assert self.state is not None, "call reset() first"
        child = self.state.children().get(int(action))
        if child is None:
            # Never applied: a masked-out action ends the episode with a penalty (a masked agent
            # cannot reach this; it keeps the unmasked ``check_env`` and random policies safe).
            return (
                self._observation(),
                INVALID_ACTION_PENALTY,
                True,
                False,
                {"invalid_action": True, **self._info()},
            )
        previous_phi = self._phi(self.state)
        self.state = child
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}
        if child.closed:
            solution = child.solution()
            terminated = True
            if solution is not None:
                self.last_solution = solution
                reward += solution.length_mm / self._reference
                info["closed"] = True
            else:  # cannot happen when the extent fits; never rewarded if it does
                info["closed"] = False
            next_phi = 0.0
        elif not child.children():
            terminated = True
            reward += DEADLOCK_PENALTY
            info["deadlock"] = True
            next_phi = 0.0
        else:
            next_phi = self._phi(child)
        reward += self.gamma * next_phi - previous_phi
        return self._observation(), float(reward), terminated, False, {**info, **self._info()}

    def action_masks(self) -> np.ndarray:
        assert self.state is not None
        mask = self.state.action_mask()
        if not mask.any():  # terminal state: keep a well-formed mask for wrappers
            mask[0] = True
        return mask

    # -- helpers ---------------------------------------------------------------------------------

    def _phi(self, state: ConstructionState) -> float:
        """Potential: reward built length, penalise the distance still to close (D28).

        Both terms are potentials, and every terminal state has potential 0, so the shaping adds
        nothing to the return of any complete episode (policy invariant) and a dead end gives back
        the length it had accumulated: an unclosed layout is never rewarded.
        """
        return (
            self.length_shaping * state.length - self.shaping * state.detour()
        ) / self._reference

    def _reward_reference(self, problem: Problem, state: ConstructionState) -> float:
        """Length that maps to a return of 1: all the rail, or the best stadium seed."""
        total = state._consts.total_length  # type: ignore[union-attr]
        if self.reward_reference == "stock":
            return total
        layouts = stadium_layouts(problem)
        if not layouts:
            return total
        seed_length = max(
            sum(local_geometry(p, m, problem.tol).length_mm for p, m in layout)
            for layout in layouts
        )
        return max(seed_length, 1.0)

    def _observation(self) -> np.ndarray:
        assert self.state is not None
        return self.state.observation()

    def _info(self) -> dict[str, Any]:
        assert self.state is not None
        return {"n_pieces": self.state.n_pieces, "length_mm": self.state.length}

    def _partial_start(self, problem: Problem) -> ConstructionState | None:
        """Curriculum: begin from a stadium seed with a window of pieces removed (a repair task)."""
        layouts = stadium_layouts(problem)
        if not layouts:
            return None
        items = layouts[int(self.np_random.integers(len(layouts)))]
        removed = int(self.np_random.integers(1, min(self.max_removed, len(items) - 1) + 1))
        try:
            return ConstructionState.from_items(
                problem, self.table, list(items[: len(items) - removed])
            )
        except ValueError:
            return None
