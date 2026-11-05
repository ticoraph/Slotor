"""Crossing -> bridge conversion and bridged seeds (D31).

A 2D crossing is two strands overlapping. A bridge lifts one strand over the other: a deck element
over the crossing, three climbing elements before it and three descending elements after it
(D30). ``convert_to_bridges`` takes a closed *flat* loop whose only fault is its crossings and
tries every way of turning them into bridges; each candidate is placed on the surface and checked
by the full validator, so only valid bridged loops come out. ``bridge_seeds`` feeds it with a
family of figure-S loops (the layout of the reference capture ``golden_b``) so that the optimizers
can start from a loop that already contains a bridge.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from shapely import STRtree

from slotor.catalog.bridges import DECK, is_raisable
from slotor.catalog.model import PieceType
from slotor.geometry import Pose, ToleranceConfig, build_track
from slotor.geometry.pieces import GeometryError, centerline_radius_mm
from slotor.optimizers.problem import Problem, Solution, finalize, fits
from slotor.validation.rules import bodies_collide, make_body

_UP = ("@u0", "@u1", "@u2")
_DOWN = ("@d2", "@d1", "@d0")
_DECK_SIZES = (1, 2)  # a deck is one element or more; two cover a crossing that straddles a joint
_MAX_BRIDGES = 3
Items = list[tuple[PieceType, bool]]


def overlapping_pairs(items: Sequence[tuple[PieceType, bool]], tol: ToleranceConfig):
    """Pairs of pieces of a chain placed from the origin that collide (level-aware)."""
    track = build_track(items, tol=tol)
    bodies = [make_body(p, tol) for p in track.pieces]
    tree = STRtree([b.shape for b in bodies])
    pairs = []
    for i, body in enumerate(bodies):
        for j in sorted(int(k) for k in tree.query(body.shape)):
            if j > i and bodies_collide(body, bodies[j], tol):
                pairs.append((i, j))
    return pairs


def _lift(problem: Problem, items: Items, deck: Sequence[int]) -> Items | None:
    """Replace the elements around ``deck`` by climb, deck and descent; ``None`` if impossible."""
    n = len(items)
    if len(deck) + 6 > n:
        return None
    first, last = deck[0], deck[-1]
    plan: dict[int, str] = {}
    for offset, suffix in zip((3, 2, 1), _UP, strict=True):
        plan[(first - offset) % n] = suffix
    for index in deck:
        plan[index % n] = DECK
    for offset, suffix in zip((1, 2, 3), _DOWN, strict=True):
        plan[(last + offset) % n] = suffix
    if len(plan) != len(deck) + 6:
        return None
    out = list(items)
    for index, suffix in plan.items():
        piece, mirrored = items[index]
        if piece.level_in or piece.level_out or not is_raisable(piece):
            return None
        out[index] = (problem.piece(piece.slug + suffix), mirrored)
    return out


def _rotate_to_ground(items: Items) -> Items:
    """Rotate the loop so that it starts on a ground-level piece (a loop starts at level 0)."""
    for index, (piece, _) in enumerate(items):
        if piece.level_in == 0 and piece.level_out == 0:
            return items[index:] + items[:index]
    return items


def convert_to_bridges(
    problem: Problem, items: Sequence[tuple[PieceType, bool]], *, limit: int = 4
) -> list[Solution]:
    """Valid bridged solutions made from a closed flat loop whose collisions are crossings.

    ``items`` is a chain of base pieces. Each crossing is resolved by lifting one of the two
    strands (trying each piece of the pair, and decks of one or two elements); resolutions are
    explored depth-first up to three bridges. Candidates are validated on the surface by
    ``finalize``; the longest ``limit`` distinct ones are returned.
    """
    start = list(items)
    found: dict[tuple, Solution] = {}

    def explore(current: Items, depth: int) -> None:
        try:
            pairs = overlapping_pairs(current, problem.tol)
        except GeometryError:
            return
        if not pairs:
            solution = finalize(problem, _rotate_to_ground(current))
            if solution is not None:
                found.setdefault(tuple(solution.slugs), solution)
            return
        if depth == _MAX_BRIDGES:
            return
        i, j = pairs[0]
        tried: set[tuple[int, ...]] = set()
        for anchor in (i, j):
            for size in _DECK_SIZES:
                for shift in range(size):
                    deck = tuple((anchor - shift + k) % len(current) for k in range(size))
                    if deck in tried:
                        continue
                    tried.add(deck)
                    lifted = _lift(problem, current, deck)
                    if lifted is not None:
                        explore(lifted, depth + 1)

    explore(start, 0)
    ranked = sorted(found.values(), key=lambda s: (-s.length_mm, s.n_pieces))
    return ranked[:limit]


# --- figure-S family: the layout of golden_b ---------------------------------------------------


def _unit_curves(piece: PieceType) -> int | None:
    """How many pieces make 60 degrees, or ``None`` when 60 is not a multiple of the angle."""
    angle = piece.turn_angle_deg or 0.0
    count = 60.0 / angle if angle else 0.0
    return int(round(count)) if count >= 1 and abs(count - round(count)) < 1e-9 else None


def _figure_s_items(curve: PieceType, per_unit: int, straight: PieceType, shape) -> Items:
    """Top row W, 180 deg left, row E, 60 right, diagonal, 120 right, bottom row W, 180 right,
    row E, 60 left, diagonal, 120 left: the S loop whose two diagonals cross."""
    a, b, c, d, e1, e2 = shape
    left, right = (curve, False), (curve, True)
    s = (straight, False)
    out: Items = []
    out += [s] * a + [left] * (3 * per_unit) + [s] * b
    out += [right] * per_unit + [s] * e1 + [right] * (2 * per_unit)
    out += [s] * c + [right] * (3 * per_unit) + [s] * d
    out += [left] * per_unit + [s] * e2 + [left] * (2 * per_unit)
    return out


def _closes(items: Items, tol: ToleranceConfig) -> bool:
    """Cheap closure test (no outlines): walk the chain with plain pose arithmetic."""
    pose = Pose(0.0, 0.0, 0.0, 0)
    for piece, mirrored in items:
        if piece.category == "straight":
            pose = pose.advance(piece.length_mm or 0.0, 0.0, 0.0)
        else:
            radius = centerline_radius_mm(piece)
            turn = math.radians(piece.turn_angle_deg or 0.0) * (-1 if mirrored else 1)
            sign = -1.0 if mirrored else 1.0
            pose = pose.advance(
                radius * math.sin(abs(turn)), sign * radius * (1 - math.cos(abs(turn))), turn
            )
    return (
        math.hypot(pose.x, pose.y) <= tol.closure_pos_mm
        and abs(math.remainder(pose.theta, 2 * math.pi)) <= 1e-6
    )


def figure_s_layouts(problem: Problem, max_rows: int = 9) -> list[Items]:
    """Closed figure-S chains that fit the inventory and surface (flat, crossings allowed)."""
    stock = problem.stock
    straight = problem.catalog.get("straight_345") if "straight_345" in stock else None
    if straight is None:
        return []
    layouts: list[Items] = []
    for curve in problem.flat_usable:
        per_unit = _unit_curves(curve) if curve.category == "curve" else None
        if per_unit is None or stock[curve.slug] < 12 * per_unit:
            continue
        for shape in itertools.product(
            range(max_rows), range(max_rows), range(max_rows), range(max_rows), (1, 2, 3), (1, 2, 3)
        ):
            if sum(shape) > stock["straight_345"]:
                continue
            items = _figure_s_items(curve, per_unit, straight, shape)
            if not _closes(items, problem.tol):
                continue
            track = build_track(items, tol=problem.tol)
            minx, miny, maxx, maxy = track.bounds
            if fits((maxx - minx, maxy - miny), problem.inner_size_mm):
                layouts.append(items)
    layouts.sort(key=lambda it: -sum(local_length(p, m, problem.tol) for p, m in it))
    return layouts


def local_length(piece: PieceType, mirrored: bool, tol: ToleranceConfig) -> float:
    from slotor.geometry import local_geometry

    return local_geometry(piece, mirrored, tol).length_mm


def bridge_seeds(problem: Problem, limit: int = 3, candidates: int = 40) -> list[Solution]:
    """Validated bridged loops from figure-S layouts, longest first (empty without bridges)."""
    if not problem.bridges:
        return []
    seeds: dict[tuple, Solution] = {}
    for items in figure_s_layouts(problem)[:candidates]:
        for solution in convert_to_bridges(problem, items, limit=1):
            seeds.setdefault(tuple(solution.slugs), solution)
        if len(seeds) >= limit:
            break
    return sorted(seeds.values(), key=lambda s: (-s.length_mm, s.n_pieces))[:limit]
