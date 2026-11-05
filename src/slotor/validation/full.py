"""The complete validator (D6): the reference every faster variant is tested against."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from shapely import STRtree

from slotor.catalog.model import PieceType
from slotor.geometry.pieces import GeometryError, LevelMismatchError
from slotor.geometry.pose import Pose
from slotor.geometry.tolerance import ToleranceConfig
from slotor.geometry.track import PlacedPiece, Track, place_piece
from slotor.validation.issues import Issue, IssueCode, Surface, ValidationResult
from slotor.validation.rules import (
    bodies_collide,
    inventory_issues,
    make_body,
    piece_issues,
    surface_issues,
)


def closure_issues(track: Track, tol: ToleranceConfig) -> list[Issue]:
    issues: list[Issue] = []
    if not track.is_closed(tol):
        err = track.closure_error()
        issues.append(
            Issue(
                IssueCode.NOT_CLOSED,
                f"loop is open: gap {err.distance_mm:.2f} mm, heading {err.dtheta:.5f} rad, "
                f"level {err.dlevel}",
            )
        )
    if track.start.level != 0 or track.exit_pose.level != 0:
        issues.append(Issue(IssueCode.NOT_AT_GROUND, "a loop must start and end at level 0"))
    return issues


def validate_layout(
    items: Sequence[tuple[PieceType, bool]],
    inventory: Mapping[str, int],
    *,
    surface: Surface | None = None,
    start: Pose | None = None,
    tol: ToleranceConfig | None = None,
    require_closed: bool = True,
    fail_fast: bool = True,
) -> ValidationResult:
    """Check inventory, connections, surface, closure and collisions, cheapest first.

    ``require_closed=False`` validates a partial chain (what optimizers build). ``fail_fast`` stops
    at the first failing stage; otherwise every reachable stage reports.
    """
    tol = ToleranceConfig() if tol is None else tol
    start = Pose(0.0, 0.0, 0.0, 0) if start is None else start
    if not items:
        issues = (Issue(IssueCode.EMPTY, "no piece"),) if require_closed else ()
        return ValidationResult(issues)

    issues: list[Issue] = []

    def stage_failed() -> bool:
        return bool(issues) and fail_fast

    # 1. Pieces and inventory.
    used: dict[str, int] = {}
    for index, (piece, _) in enumerate(items):
        issues += piece_issues(piece, index)
        for key in piece.inventory_keys:
            used[key] = used.get(key, 0) + 1
    issues += inventory_issues(used, inventory)
    if stage_failed():
        return ValidationResult(tuple(issues))

    # 2. Connections: geometry known, levels consistent.
    placed: list[PlacedPiece] = []
    pose = start
    for index, (piece, mirrored) in enumerate(items):
        try:
            item = place_piece(pose, piece, mirrored, tol, index)
        except LevelMismatchError as exc:
            issues.append(Issue(IssueCode.LEVEL_MISMATCH, str(exc), (index,)))
            return ValidationResult(tuple(issues))
        except GeometryError as exc:
            issues.append(Issue(IssueCode.GEOMETRY, str(exc), (index,)))
            return ValidationResult(tuple(issues))
        placed.append(item)
        pose = item.exit
    track = Track(start, tuple(placed))

    # 3. Surface.
    for item in placed:
        issues += surface_issues(item, surface, tol)
    if stage_failed():
        return ValidationResult(tuple(issues), track)

    # 4. Closure.
    if require_closed:
        issues += closure_issues(track, tol)
        if stage_failed():
            return ValidationResult(tuple(issues), track)

    # 5. Collisions: STRtree pre-filter, then the area test.
    bodies = [make_body(item, tol) for item in placed]
    tree = STRtree([body.shape for body in bodies])
    for i, body in enumerate(bodies):
        for j in sorted(int(k) for k in tree.query(body.shape)):
            if j > i and bodies_collide(body, bodies[j], tol):
                issues.append(Issue(IssueCode.COLLISION, f"pieces {i} and {j} overlap", (i, j)))
    return ValidationResult(tuple(issues), track)
