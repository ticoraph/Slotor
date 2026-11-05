"""Construction state shared by the RL environment and learned priors (formulations a, b, c).

It owns no physics: every placement goes through the incremental validator (D6). The search runs
in free space from the pose ``(0, 0, 0)`` like the Beam Search; the surface only bounds the extent
of the layout (as is or turned a quarter turn), and a closed loop is turned into a ``Solution``
by ``finalize`` (full validator).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from slotor.catalog.model import Catalog, PieceType
from slotor.geometry import Pose, local_geometry
from slotor.geometry.pose import wrap_angle
from slotor.optimizers.beam import can_still_close
from slotor.optimizers.problem import SUPPORTED_CATEGORIES, Problem, Solution, finalize, fits
from slotor.validation import IncrementalValidator

ROOT = Pose(0.0, 0.0, 0.0, 0)
OBS_SIZE = 22
_LENGTH_SCALE = 1500.0  # mm, scale of relative positions in the observation
_MAX_COUNT = 16.0


@dataclass(frozen=True)
class ActionTable:
    """Fixed, versioned action space: one index per ``(piece, mirrored)`` of the catalog.

    The table depends on the catalog only (never on the inventory), so action indices stay
    stable across instances; the mask carries the instance-specific availability.
    """

    entries: tuple[tuple[str, bool], ...]
    slugs: tuple[str, ...]
    catalog_sha256: str

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> ActionTable:
        pieces = sorted(
            (p for p in catalog.pieces if p.verified and p.category in SUPPORTED_CATEGORIES),
            key=lambda p: p.slug,
        )
        entries = []
        for piece in pieces:
            entries.append((piece.slug, False))
            if piece.category == "curve":
                entries.append((piece.slug, True))
        return cls(tuple(entries), tuple(p.slug for p in pieces), catalog.content_sha256)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class ConstructionState:
    problem: Problem
    table: ActionTable
    validator: IncrementalValidator
    length: float
    extent: tuple[float, float, float, float] | None
    closed: bool = False
    _children: dict[int, ConstructionState] | None = field(default=None, repr=False)
    _consts: _Consts | None = field(default=None, repr=False)

    @classmethod
    def initial(cls, problem: Problem, table: ActionTable) -> ConstructionState:
        validator = IncrementalValidator(problem.inventory, start=ROOT, tol=problem.tol)
        state = cls(problem, table, validator, 0.0, None)
        state._consts = _Consts.of(problem)
        return state

    @classmethod
    def from_items(
        cls, problem: Problem, table: ActionTable, items: list[tuple[PieceType, bool]]
    ) -> ConstructionState:
        """A state holding an existing chain (free space, start at ROOT); it must be valid."""
        state = cls.initial(problem, table)
        for piece, mirrored in items:
            index = table.entries.index((piece.slug, mirrored))
            child = state.children().get(index)
            if child is None:
                raise ValueError(f"item {piece.slug!r} cannot be placed at this point")
            state = child
        return state

    @property
    def pose(self) -> Pose:
        return self.validator.end_pose

    @property
    def n_pieces(self) -> int:
        return len(self.validator)

    def _derive(
        self, child_validator: IncrementalValidator, piece_length: float
    ) -> ConstructionState:
        placed = child_validator.placed[-1]
        box = placed.outline.bounds
        extent = (
            box
            if self.extent is None
            else (
                min(self.extent[0], box[0]),
                min(self.extent[1], box[1]),
                max(self.extent[2], box[2]),
                max(self.extent[3], box[3]),
            )
        )
        child = ConstructionState(
            self.problem, self.table, child_validator, self.length + piece_length, extent
        )
        child._consts = self._consts
        return child

    def children(self) -> dict[int, ConstructionState]:
        """Valid successors by action index (empty once closed). Cached: the mask reuses them."""
        if self._children is not None:
            return self._children
        children: dict[int, ConstructionState] = {}
        if not self.closed:
            consts = self._consts
            assert consts is not None
            used = self.validator.used
            for index, (slug, mirrored) in enumerate(self.table.entries):
                stock = consts.stock.get(slug, 0)
                if used.get(slug, 0) >= stock:
                    continue
                piece = consts.pieces[slug]
                child_validator = self.validator.fork()
                if child_validator.add(piece, mirrored):
                    continue
                child = self._derive(child_validator, consts.length[slug])
                assert child.extent is not None
                if not fits(
                    (child.extent[2] - child.extent[0], child.extent[3] - child.extent[1]),
                    consts.inner,
                ):
                    continue
                if _closed_at_root(child.pose, self.problem):
                    if child.validator.result().valid:
                        child.closed = True
                        children[index] = child
                    continue
                left_length, left_turn = consts.remaining(child.validator.used)
                if can_still_close(
                    child.pose, ROOT, left_length, left_turn, self.problem.tol, consts.min_width
                ):
                    children[index] = child
        self._children = children
        return children

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.table), dtype=bool)
        for index in self.children():
            mask[index] = True
        return mask

    def solution(self) -> Solution | None:
        """The validated solution of a closed state (full validator, on the surface)."""
        return finalize(self.problem, self.validator.items) if self.closed else None

    def detour(self) -> float:
        """Distance and heading gap to the closure target, as one length (potential function)."""
        pose = self.pose
        return math.hypot(pose.x, pose.y) + 297.0 * abs(wrap_angle(pose.theta))

    def observation(self) -> np.ndarray:
        consts = self._consts
        assert consts is not None
        used = self.validator.used
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        for i, slug in enumerate(self.table.slugs):
            obs[i] = min((consts.stock.get(slug, 0) - used.get(slug, 0)) / _MAX_COUNT, 1.0)
        pose = self.pose
        # Position of the start, seen from the current pose (frame of the current connector).
        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        dx, dy = -pose.x, -pose.y
        obs[8] = np.clip((dx * cos_t + dy * sin_t) / _LENGTH_SCALE, -2.0, 2.0)
        obs[9] = np.clip((-dx * sin_t + dy * cos_t) / _LENGTH_SCALE, -2.0, 2.0)
        gap = wrap_angle(-pose.theta)
        obs[10], obs[11] = math.sin(gap), math.cos(gap)
        if self.extent is not None:
            width, height = self.extent[2] - self.extent[0], self.extent[3] - self.extent[1]
            long_side, short_side = max(width, height), min(width, height)
            obs[12] = long_side / max(consts.inner)
            obs[13] = short_side / min(consts.inner)
        obs[14] = consts.inner[0] / 3000.0
        obs[15] = consts.inner[1] / 2000.0
        obs[16] = self.length / consts.total_length
        obs[17] = self.n_pieces / 40.0
        left_length, left_turn = consts.remaining(used)
        obs[18] = left_length / consts.total_length
        obs[19] = min(left_turn / 360.0, 4.0)
        obs[20], obs[21] = math.sin(pose.theta), math.cos(pose.theta)
        return obs


@dataclass(frozen=True)
class _Consts:
    """Per-problem constants, computed once and shared by every state of an episode."""

    stock: dict[str, int]
    pieces: dict[str, PieceType]
    length: dict[str, float]
    turn: dict[str, float]
    inner: tuple[float, float]
    total_length: float
    min_width: float

    @classmethod
    def of(cls, problem: Problem) -> _Consts:
        if problem.bridges:
            raise ValueError("the RL environment supports flat problems only (no bridges)")
        pieces = {p.slug: p for p in problem.usable}
        geometry = {slug: local_geometry(p, False, problem.tol) for slug, p in pieces.items()}
        stock = problem.stock
        length = {slug: g.length_mm for slug, g in geometry.items()}
        return cls(
            stock=stock,
            pieces=pieces,
            length=length,
            turn={slug: (p.turn_angle_deg or 0.0) for slug, p in pieces.items()},
            inner=problem.inner_size_mm,
            total_length=max(sum(length[s] * n for s, n in stock.items()), 1.0),
            min_width=min(g.width_mm for g in geometry.values()),
        )

    def remaining(self, used: dict[str, int]) -> tuple[float, float]:
        left_length = sum(
            self.length[s] * max(n - used.get(s, 0), 0) for s, n in self.stock.items()
        )
        left_turn = sum(self.turn[s] * max(n - used.get(s, 0), 0) for s, n in self.stock.items())
        return left_length, left_turn


def _closed_at_root(pose: Pose, problem: Problem) -> bool:
    tol = problem.tol
    return (
        math.hypot(pose.x, pose.y) <= 2 * tol.closure_pos_mm and abs(wrap_angle(pose.theta)) <= 1e-2
    )
