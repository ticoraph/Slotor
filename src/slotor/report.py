"""Presentation data for a solution: figures, bill of materials, JSON and SVG exports.

Everything the UI shows is computed here, so the UI itself stays free of business logic (D2).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

from slotor.catalog.bridges import DECK_LEVEL
from slotor.geometry import Pose
from slotor.optimizers.problem import Problem, Solution, make_solution

FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class UsageRow:
    slug: str
    name: str
    used: int
    available: int

    @property
    def left(self) -> int:
        return self.available - self.used


@dataclass(frozen=True)
class SolutionReport:
    length_mm: float
    n_pieces: int
    extent_mm: tuple[float, float]  # bounding box of the layout
    surface_mm: tuple[float, float, float]  # width, height, margin
    usage: tuple[UsageRow, ...]
    lane_spacing_mm: float | None
    lane_lengths_mm: tuple[float, float] | None  # left lane, right lane (direction of travel)
    ignored: dict[str, int]  # inventory the optimizers cannot use yet
    parts: tuple[tuple[str, str, int], ...] = ()  # (slug, name, count) of what is laid, by variant
    elevated: dict[str, int] = field(
        default_factory=dict
    )  # bridges / ramp elements / raised elements


def build_report(
    solution: Solution, problem: Problem, lane_spacing_mm: float | None = None
) -> SolutionReport:
    track = solution.validation.track
    assert track is not None
    minx, miny, maxx, maxy = track.bounds
    used = Counter(piece.inventory_key for piece, _ in solution.items)
    laid = Counter((piece.slug, piece.name) for piece, _ in solution.items)
    elevated = {"bridges": 0, "raised": 0, "ramp": 0}
    for piece, _ in solution.items:
        if piece.level_in == piece.level_out > 0:
            elevated["raised"] += 1
        elif piece.level_in != piece.level_out:
            elevated["ramp"] += 1
            if (piece.level_in, piece.level_out) == (2, DECK_LEVEL):
                elevated["bridges"] += (
                    1  # the third element of a climb: one bridge reaches its deck
                )
    rows = []
    for slug in sorted(set(used) | {s for s, n in problem.inventory.items() if n > 0}):
        if slug.startswith("@"):
            continue
        if slug in problem.ignored:
            continue
        name = problem.catalog.get(slug).name if slug in problem.catalog else slug
        rows.append(UsageRow(slug, name, used.get(slug, 0), problem.inventory.get(slug, 0)))
    lanes = None
    if lane_spacing_mm is not None:
        half = lane_spacing_mm / 2
        lanes = (track.lane_length_mm(+half), track.lane_length_mm(-half))
    width, height, margin = (
        problem.surface.width_mm,
        problem.surface.height_mm,
        problem.surface.margin_mm,
    )
    return SolutionReport(
        length_mm=solution.length_mm,
        n_pieces=solution.n_pieces,
        extent_mm=(maxx - minx, maxy - miny),
        surface_mm=(width, height, margin),
        usage=tuple(rows),
        lane_spacing_mm=lane_spacing_mm,
        lane_lengths_mm=lanes,
        ignored=dict(problem.ignored),
        parts=tuple(sorted((slug, name, n) for (slug, name), n in laid.items())),
        elevated=elevated if any(elevated.values()) else {},
    )


def format_summary(report: SolutionReport) -> str:
    """Markdown summary shown next to the drawing."""
    width, height = report.extent_mm
    lines = [
        f"**Longueur de la ligne médiane : {report.length_mm / 1000:.3f} m** "
        f"({report.length_mm:.0f} mm), {report.n_pieces} pièces.",
        f"Emprise du circuit : {width:.0f} × {height:.0f} mm "
        f"(surface {report.surface_mm[0]:.0f} × {report.surface_mm[1]:.0f} mm, "
        f"marge {report.surface_mm[2]:.0f} mm).",
    ]
    if report.lane_lengths_mm is None:
        lines.append(
            "Longueur des deux voies : non calculée (entraxe des fentes inconnu, "
            "à renseigner ou à mesurer)."
        )
    else:
        left, right = report.lane_lengths_mm
        lines.append(
            f"Voie gauche : {left / 1000:.3f} m, voie droite : {right / 1000:.3f} m "
            f"(entraxe {report.lane_spacing_mm:.1f} mm, écart {abs(left - right):.0f} mm)."
        )
    if report.elevated:
        e = report.elevated
        lines.append(
            f"Pont : {e['bridges']} pont(s), {e['ramp']} éléments de rampe, "
            f"{e['raised']} de tablier. "
            "Supports à prévoir sous tous ces éléments ; hauteur, pente et dégagement ne sont pas "
            "vérifiés."
        )
    if report.ignored:
        ignored = ", ".join(f"{slug} × {n}" for slug, n in report.ignored.items())
        lines.append(f"Pièces ignorées (pas encore modélisées) : {ignored}.")
    return "\n\n".join(lines)


def bill_of_materials(report: SolutionReport) -> str:
    """Plain-text parts list to assemble the loop."""
    lines = [f"{count} × {name} ({slug})" for slug, name, count in report.parts]
    lines.append(f"Total : {report.n_pieces} pièces")
    return "\n".join(lines)


def usage_rows(report: SolutionReport) -> list[list[object]]:
    return [[r.slug, r.name, r.used, r.available, r.left] for r in report.usage]


USAGE_HEADERS = ["Référence", "Pièce", "Utilisées", "Disponibles", "Restantes"]


def solution_to_json(
    solution: Solution, problem: Problem, lane_spacing_mm: float | None = None
) -> str:
    track = solution.validation.track
    assert track is not None
    report = build_report(solution, problem, lane_spacing_mm)
    data = {
        "format": "slotor-solution",
        "version": FORMAT_VERSION,
        "catalog": {"name": problem.catalog.name, "sha256": problem.catalog.content_sha256},
        "surface_mm": {
            "width": problem.surface.width_mm,
            "height": problem.surface.height_mm,
            "margin": problem.surface.margin_mm,
        },
        "inventory": dict(sorted(problem.inventory.items())),
        "bridges": problem.bridges,
        "centerline_length_mm": solution.length_mm,
        "lane_spacing_mm": lane_spacing_mm,
        "lane_lengths_mm": list(report.lane_lengths_mm) if report.lane_lengths_mm else None,
        "start": {
            "x": solution.start.x,
            "y": solution.start.y,
            "theta_deg": math.degrees(solution.start.theta),
            "level": solution.start.level,
        },
        "pieces": [
            {
                "slug": placed.geometry.slug,
                "mirrored": mirrored,
                "entry": {
                    "x": placed.entry.x,
                    "y": placed.entry.y,
                    "theta_deg": math.degrees(placed.entry.theta),
                },
            }
            for placed, (_, mirrored) in zip(track.pieces, solution.items, strict=True)
        ],
    }
    return json.dumps(data, indent=2)


def solution_from_json(text: str, problem: Problem) -> Solution:
    """Rebuild a solution from its export and re-validate it: a file is never trusted."""
    try:
        data = json.loads(text)
        if data.get("format") != "slotor-solution":
            raise ValueError("not a slotor solution file")
        start = data["start"]
        pose = Pose(float(start["x"]), float(start["y"]), math.radians(start["theta_deg"]), 0)
        items = tuple(
            (problem.piece(piece["slug"]), bool(piece["mirrored"])) for piece in data["pieces"]
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed solution file: {exc}") from exc
    solution = make_solution(problem, items, pose)
    if solution is None:
        raise ValueError("the stored layout is not a valid closed loop for this problem")
    return solution


_FILL = {"straight": "#5a5a60", "curve": "#6f7c99"}
_LEVEL_FILL = {1: "#a07a3a", 2: "#c9a15a"}  # ramps, raised pieces


def solution_to_svg(solution: Solution, problem: Problem, *, width_px: float = 900.0) -> str:
    """Drawing to scale of the surface and the loop (y axis up, millimetres, 100 mm grid)."""
    surface = problem.surface
    scale = width_px / surface.width_mm
    height_px = surface.height_mm * scale

    def point(x: float, y: float) -> str:
        return f"{x * scale:.2f},{(surface.height_mm - y) * scale:.2f}"

    track = solution.validation.track
    assert track is not None
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px:.0f}" height="{height_px:.0f}" '
        f'viewBox="0 0 {width_px:.0f} {height_px:.0f}">',
        '<rect width="100%" height="100%" fill="#101216"/>',
    ]
    for x in range(0, int(surface.width_mm) + 1, 100):
        parts.append(
            f'<line x1="{x * scale:.1f}" y1="0" x2="{x * scale:.1f}" y2="{height_px:.0f}" '
            'stroke="#23272e" stroke-width="0.5"/>'
        )
    for y in range(0, int(surface.height_mm) + 1, 100):
        parts.append(
            f'<line x1="0" y1="{y * scale:.1f}" x2="{width_px:.0f}" y2="{y * scale:.1f}" '
            'stroke="#23272e" stroke-width="0.5"/>'
        )
    left, bottom, right, top = surface.bounds
    parts.append(
        f'<rect x="{left * scale:.1f}" y="{(surface.height_mm - top) * scale:.1f}" '
        f'width="{(right - left) * scale:.1f}" height="{(top - bottom) * scale:.1f}" '
        'fill="none" stroke="#c9a227" stroke-width="1" stroke-dasharray="6 4"/>'
    )

    # Ground first, then ramps, then raised pieces on top, so a bridge reads as a bridge.
    def layer(placed) -> int:
        if placed.entry.level == placed.exit.level > 0:
            return 2
        return 1 if placed.entry.level != placed.exit.level else 0

    for placed in sorted(track.pieces, key=lambda p: (layer(p), p.index)):
        coords = " ".join(point(x, y) for x, y in placed.outline.exterior.coords)
        level = layer(placed)
        fill = _FILL.get(placed.geometry.kind, "#666666") if level == 0 else _LEVEL_FILL[level]
        stroke = "#d0d0d0" if level == 0 else "#ffd166"
        width = "0.6" if level == 0 else "1.4"
        title = escape(f"{placed.index + 1}. {placed.geometry.slug}")
        parts.append(
            f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" stroke-width="{width}">'
            f"<title>{title}</title></polygon>"
        )
    first = track.pieces[0].entry
    sx, sy = point(first.x, first.y).split(",")
    parts.append(f'<circle cx="{sx}" cy="{sy}" r="5" fill="#e03030"><title>départ</title></circle>')
    label = escape(
        f"{surface.width_mm:.0f} × {surface.height_mm:.0f} mm — "
        f"{solution.length_mm / 1000:.2f} m — {solution.n_pieces} pièces"
    )
    parts.append(
        f'<text x="8" y="16" fill="#e8e8e8" font-size="13" font-family="sans-serif">{label}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)
