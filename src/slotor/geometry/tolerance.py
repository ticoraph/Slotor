"""Numerical tolerances (D3, D8, D9): named parameters, never magic constants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToleranceConfig:
    """Tolerances of the geometry engine.

    Closure play comes from the user's hands-on experience (D9, D23), not from a measurement yet
    (sheet E1): a loop closes with up to 10 mm of position gap, and an angular misalignment of at
    most 1 mm. The angle is expressed as a displacement at the rail edge, so the allowed angle is
    ``closure_angle_edge_mm / width`` (about 5e-3 rad, 0.29 deg, for a 198 mm rail). Revisit both
    after E1. Piece geometry itself is ideal, so an exactly closed loop closes to float rounding.
    """

    closure_pos_mm: float = 10.0
    closure_angle_edge_mm: float = 1.0
    # Maximum distance between an arc and the chord used to draw it in an outline polygon.
    arc_sagitta_mm: float = 0.05
    # Collision (D8, D24): two outlines collide when their overlap, after each is eroded by half of
    # ``collision_depth_mm``, exceeds ``collision_area_mm2``. Calibrated on golden_a, whose rails
    # interpenetrate by less than 1 mm because of its 1.19 mm closure gap; provisional (sheet E1).
    collision_depth_mm: float = 2.0
    collision_area_mm2: float = 1.0
    # Numeric slack for 'inside the surface' (a piece may sit exactly on the margin line).
    surface_tol_mm: float = 1e-3

    def __post_init__(self) -> None:
        for name in (
            "closure_pos_mm",
            "closure_angle_edge_mm",
            "arc_sagitta_mm",
            "collision_depth_mm",
            "collision_area_mm2",
            "surface_tol_mm",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
