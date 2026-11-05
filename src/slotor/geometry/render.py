"""Debug rendering: a standalone SVG of a track (stdlib only, no business logic)."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from slotor.geometry.track import Track

_PALETTE = ("#4a4a4f", "#5d6a85")


def track_to_svg(track: Track, *, margin_mm: float = 50.0, px_per_mm: float = 0.4) -> str:
    """Return an SVG document; the y axis is flipped so that left turns look counter-clockwise."""
    minx, miny, maxx, maxy = track.bounds
    width = (maxx - minx + 2 * margin_mm) * px_per_mm
    height = (maxy - miny + 2 * margin_mm) * px_per_mm

    def point(x: float, y: float) -> str:
        return f"{(x - minx + margin_mm) * px_per_mm:.2f},{(maxy - y + margin_mm) * px_per_mm:.2f}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}">',
        '<rect width="100%" height="100%" fill="#101010"/>',
    ]
    for placed in track.pieces:
        coords = " ".join(point(x, y) for x, y in placed.outline.exterior.coords)
        fill = _PALETTE[placed.index % 2]
        label = escape(f"{placed.index}: {placed.geometry.slug}")
        parts.append(
            f'<polygon points="{coords}" fill="{fill}" stroke="#bbbbbb" stroke-width="0.6">'
            f"<title>{label}</title></polygon>"
        )
    start = track.start
    parts.append(
        f'<circle cx="{point(start.x, start.y).split(",")[0]}" '
        f'cy="{point(start.x, start.y).split(",")[1]}" r="4" fill="#e03030"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(track: Track, path: str | Path, **kwargs: float) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(track_to_svg(track, **kwargs), encoding="utf-8")
    return target
