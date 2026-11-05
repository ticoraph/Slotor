"""Rigid poses ``(x, y, theta, level)``: millimetres, radians, ``level in {0, 1}`` (D3, D4)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pose:
    x: float
    y: float
    theta: float
    level: int = 0

    def advance(self, dx: float, dy: float, dtheta: float, level: int | None = None) -> Pose:
        """Compose with a local displacement expressed in this pose's own frame."""
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        return Pose(
            self.x + dx * cos_t - dy * sin_t,
            self.y + dx * sin_t + dy * cos_t,
            self.theta + dtheta,
            self.level if level is None else level,
        )

    def compose(self, local: Pose) -> Pose:
        """Express a pose given in this pose's frame in the parent frame."""
        return self.advance(local.x, local.y, local.theta, local.level)


def wrap_angle(angle: float) -> float:
    """Wrap an angle into ``(-pi, pi]``."""
    wrapped = math.remainder(angle, 2 * math.pi)
    return math.pi if wrapped == -math.pi else wrapped
