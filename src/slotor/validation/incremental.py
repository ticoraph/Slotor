"""Incremental validator: check one added piece without recomputing the chain (D6).

It reuses the rule functions of the full validator and is tested against it. Optimizers use
``check_add``/``add``/``pop``; the verdict on a finished loop comes from ``result``.
"""

from __future__ import annotations

from collections.abc import Mapping

from shapely import STRtree

from slotor.catalog.model import PieceType
from slotor.geometry.pieces import GeometryError, LevelMismatchError
from slotor.geometry.pose import Pose
from slotor.geometry.tolerance import ToleranceConfig
from slotor.geometry.track import PlacedPiece, Track, place_piece
from slotor.validation.full import closure_issues
from slotor.validation.issues import Issue, IssueCode, Surface, ValidationResult
from slotor.validation.rules import (
    Body,
    bodies_collide,
    inventory_issues,
    make_body,
    piece_issues,
    surface_issues,
)


class IncrementalValidator:
    def __init__(
        self,
        inventory: Mapping[str, int],
        *,
        surface: Surface | None = None,
        start: Pose | None = None,
        tol: ToleranceConfig | None = None,
    ) -> None:
        self._inventory = dict(inventory)
        self._surface = surface
        self._start = Pose(0.0, 0.0, 0.0, 0) if start is None else start
        self._tol = ToleranceConfig() if tol is None else tol
        self._items: list[tuple[PieceType, bool]] = []
        self._placed: list[PlacedPiece] = []
        self._bodies: list[Body] = []
        self._used: dict[str, int] = {}
        self._tree: STRtree | None = None

    def __len__(self) -> int:
        return len(self._placed)

    @property
    def items(self) -> tuple[tuple[PieceType, bool], ...]:
        return tuple(self._items)

    @property
    def end_pose(self) -> Pose:
        return self._placed[-1].exit if self._placed else self._start

    @property
    def used(self) -> Mapping[str, int]:
        return dict(self._used)

    @property
    def placed(self) -> tuple[PlacedPiece, ...]:
        return tuple(self._placed)

    def fork(self) -> IncrementalValidator:
        """Independent copy for tree search; shares the immutable pieces, bodies and index."""
        other = IncrementalValidator.__new__(IncrementalValidator)
        other._inventory = self._inventory
        other._surface = self._surface
        other._start = self._start
        other._tol = self._tol
        other._items = list(self._items)
        other._placed = list(self._placed)
        other._bodies = list(self._bodies)
        other._used = dict(self._used)
        other._tree = self._tree
        return other

    def track(self) -> Track:
        return Track(self._start, tuple(self._placed))

    def _candidate(
        self, piece: PieceType, mirrored: bool
    ) -> tuple[list[Issue], PlacedPiece | None]:
        index = len(self._placed)
        issues = piece_issues(piece, index)
        issues += inventory_issues(
            {key: self._used.get(key, 0) + 1 for key in piece.inventory_keys}, self._inventory
        )
        if issues:
            return issues, None
        try:
            placed = place_piece(self.end_pose, piece, mirrored, self._tol, index)
        except LevelMismatchError as exc:
            return [Issue(IssueCode.LEVEL_MISMATCH, str(exc), (index,))], None
        except GeometryError as exc:
            return [Issue(IssueCode.GEOMETRY, str(exc), (index,))], None
        issues += surface_issues(placed, self._surface, self._tol)
        if issues:
            return issues, placed
        return self._collision_issues(placed), placed

    def _collision_issues(self, placed: PlacedPiece) -> list[Issue]:
        if not self._bodies:
            return []
        body = make_body(placed, self._tol)
        if self._tree is None:
            self._tree = STRtree([b.shape for b in self._bodies])
        hits = sorted(int(k) for k in self._tree.query(body.shape))
        return [
            Issue(IssueCode.COLLISION, f"pieces {j} and {placed.index} overlap", (j, placed.index))
            for j in hits
            if bodies_collide(self._bodies[j], body, self._tol)
        ]

    def check_add(self, piece: PieceType, mirrored: bool = False) -> tuple[Issue, ...]:
        """Issues the piece would cause; empty means it can be added. Does not change the state."""
        return tuple(self._candidate(piece, mirrored)[0])

    def add(self, piece: PieceType, mirrored: bool = False) -> tuple[Issue, ...]:
        """Add the piece when valid and return no issue; otherwise change nothing and report."""
        issues, placed = self._candidate(piece, mirrored)
        if issues or placed is None:
            return tuple(issues)
        self._items.append((piece, mirrored))
        self._placed.append(placed)
        self._bodies.append(make_body(placed, self._tol))
        for key in piece.inventory_keys:
            self._used[key] = self._used.get(key, 0) + 1
        self._tree = None
        return ()

    def pop(self) -> None:
        if not self._placed:
            raise IndexError("pop from an empty layout")
        piece, _ = self._items.pop()
        self._placed.pop()
        self._bodies.pop()
        for key in piece.inventory_keys:
            self._used[key] -= 1
            if not self._used[key]:
                del self._used[key]
        self._tree = None

    def result(self, *, require_closed: bool = True) -> ValidationResult:
        """Verdict on the current chain; every stored piece already passed ``add``."""
        if not self._placed:
            issues = (Issue(IssueCode.EMPTY, "no piece"),) if require_closed else ()
            return ValidationResult(issues)
        track = self.track()
        issues = closure_issues(track, self._tol) if require_closed else []
        return ValidationResult(tuple(issues), track)
