"""Closed seeds (D10): stadiums built from the inventory, then validated like any other layout.

A stadium is two half-turns of the same centerline radius joined by two straight sides of equal
length. The sides may be composed differently (e.g. 2 x 345 on one, 345 + 3 x 115 on the other).
Each half-turn is a multiset of curves of one radius family summing to 180 degrees.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

from slotor.catalog.model import PieceType
from slotor.geometry.pieces import centerline_radius_mm
from slotor.geometry.track import build_track
from slotor.optimizers.problem import Problem, Solution, finalize, fits

_ANGLE_UNIT_DEG = 0.001  # integer angle arithmetic keeps half-turn sums exact (D9)
_LENGTH_UNIT_MM = 0.01
HALF_TURN_DEG = 180.0


def _half_turn_compositions(
    family: list[PieceType], stock: dict[str, int]
) -> list[tuple[tuple[PieceType, ...], ...]]:
    """Every multiset of ``family`` curves that makes 180 degrees, as lists of pieces."""
    target = round(HALF_TURN_DEG / _ANGLE_UNIT_DEG)
    angles = [round(p.turn_angle_deg / _ANGLE_UNIT_DEG) for p in family]
    found: list[tuple[tuple[PieceType, ...], ...]] = []

    def recurse(index: int, remaining: int, chosen: list[int]) -> None:
        if index == len(family):
            if remaining == 0:
                found.append(
                    tuple(tuple([family[i]] * chosen[i]) for i in range(len(family)) if chosen[i])
                )
            return
        most = min(stock[family[index].slug], remaining // angles[index])
        for count in range(most + 1):
            recurse(index + 1, remaining - count * angles[index], [*chosen, count])

    recurse(0, target, [])
    return found


def _flatten(groups: tuple[tuple[PieceType, ...], ...]) -> list[PieceType]:
    return [piece for group in groups for piece in group]


def _count(pieces: list[PieceType]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for piece in pieces:
        counts[piece.slug] += 1
    return counts


def _side_pairs(
    straights: list[PieceType], stock: dict[str, int], used_by_curves: dict[str, int]
) -> list[tuple[float, list[PieceType], list[PieceType]]]:
    """Pairs of disjoint straight multisets of equal length, longest first."""
    available = [stock[p.slug] - used_by_curves.get(p.slug, 0) for p in straights]
    by_length: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for counts in itertools.product(*(range(n + 1) for n in available)):
        length = sum(c * p.length_mm for c, p in zip(counts, straights, strict=True))
        by_length[round(length / _LENGTH_UNIT_MM)].append(counts)
    pairs = []
    for key, vectors in by_length.items():
        for a, b in itertools.combinations_with_replacement(vectors, 2):
            if all(x + y <= n for x, y, n in zip(a, b, available, strict=True)):
                pairs.append((key * _LENGTH_UNIT_MM, a, b))
    pairs.sort(key=lambda t: -t[0])

    def expand(counts: tuple[int, ...]) -> list[PieceType]:
        return [p for c, p in zip(counts, straights, strict=True) for _ in range(c)]

    return [(length, expand(a), expand(b)) for length, a, b in pairs]


def stadium_layouts(problem: Problem) -> list[list[tuple[PieceType, bool]]]:
    """Candidate stadium layouts, longest first, each fitting the surface in free space."""
    usable = problem.flat_usable
    stock = problem.stock
    curves = [p for p in usable if p.category == "curve"]
    straights = [p for p in usable if p.category == "straight"]
    families: dict[float, list[PieceType]] = defaultdict(list)
    for piece in curves:
        families[round(centerline_radius_mm(piece), 6)].append(piece)

    candidates: list[tuple[float, list[tuple[PieceType, bool]]]] = []
    for family in families.values():
        compositions = _half_turn_compositions(family, stock)
        for first, second in itertools.combinations_with_replacement(compositions, 2):
            turns = _flatten(first) + _flatten(second)
            curve_use = _count(turns)
            if any(curve_use[s] > stock[s] for s in curve_use):
                continue
            for _length, side_a, side_b in _side_pairs(straights, stock, curve_use):
                items = (
                    [(p, False) for p in _flatten(first)]
                    + [(p, False) for p in side_a]
                    + [(p, False) for p in _flatten(second)]
                    + [(p, False) for p in side_b]
                )
                track = build_track(items, tol=problem.tol)
                minx, miny, maxx, maxy = track.bounds
                if not fits((maxx - minx, maxy - miny), problem.inner_size_mm):
                    continue
                candidates.append((track.length_mm, items))
                break  # sides are sorted longest first: the first that fits is the best here
    candidates.sort(key=lambda t: -t[0])
    return [items for _, items in candidates]


def generate_seeds(problem: Problem, limit: int | None = None) -> list[Solution]:
    """Validated closed stadiums, longest first. Empty when the inventory makes none."""
    seeds: list[Solution] = []
    for items in stadium_layouts(problem):
        solution = finalize(problem, items)
        if solution is not None:
            seeds.append(solution)
        if limit is not None and len(seeds) >= limit:
            break
    seeds.sort(key=lambda s: (-s.length_mm, s.n_pieces))
    return seeds
