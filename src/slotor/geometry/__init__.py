"""Geometry engine: poses, piece outlines, chains of pieces, closure and lane lengths."""

from slotor.geometry.pieces import GeometryError, LevelMismatchError, LocalGeometry, local_geometry
from slotor.geometry.pose import Pose, wrap_angle
from slotor.geometry.render import track_to_svg, write_svg
from slotor.geometry.tolerance import ToleranceConfig
from slotor.geometry.track import ClosureError, PlacedPiece, Track, build_track, place_piece

__all__ = [
    "ClosureError",
    "GeometryError",
    "LevelMismatchError",
    "LocalGeometry",
    "PlacedPiece",
    "Pose",
    "ToleranceConfig",
    "Track",
    "build_track",
    "local_geometry",
    "place_piece",
    "track_to_svg",
    "wrap_angle",
    "write_svg",
]
