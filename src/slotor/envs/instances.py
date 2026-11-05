"""Instance distributions for the RL environment and the frozen evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from slotor.catalog.model import Catalog
from slotor.optimizers.problem import Problem
from slotor.validation import Surface

# Domain randomization (documented in D28): surface and inventory drawn at every ``reset`` so the
# agent cannot memorise one instance. Probability that an optional piece type is present, and its
# count range. The evaluation suite never comes from this sampler's training seeds.
_REQUIRED = {"curve_r1_60": (4, 14), "straight_345": (2, 16)}
_OPTIONAL = {
    "curve_r1_30": (0.4, 1, 6),
    "curve_r2_30": (0.3, 1, 8),
    "curve_r3_30": (0.15, 1, 6),
    "curve_r4_15": (0.1, 1, 12),
    "straight_115": (0.5, 1, 6),
    "straight_86_25": (0.25, 1, 3),
}
SURFACE_RANGE_MM = ((1500, 3200), (800, 2000))


class InstanceSampler:
    """Random problems with a documented distribution."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def sample(self, rng: np.random.Generator) -> Problem:
        (w_lo, w_hi), (h_lo, h_hi) = SURFACE_RANGE_MM
        width = 10 * int(rng.integers(w_lo // 10, w_hi // 10 + 1))
        height = 10 * int(rng.integers(h_lo // 10, h_hi // 10 + 1))
        inventory = {slug: int(rng.integers(lo, hi + 1)) for slug, (lo, hi) in _REQUIRED.items()}
        for slug, (probability, lo, hi) in _OPTIONAL.items():
            if rng.random() < probability:
                inventory[slug] = int(rng.integers(lo, hi + 1))
        return Problem(self.catalog, inventory, Surface(float(width), float(height)))


def load_suite(
    path: str | Path, catalog: Catalog, split: str | None = None
) -> list[tuple[str, Problem]]:
    """Frozen instances from ``{"instances": [{name, split?, surface_mm, inventory}]}`` (JSON)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    suite = []
    for spec in data["instances"]:
        if split is not None and spec.get("split", "test") != split:
            continue
        width, height = spec["surface_mm"]
        problem = Problem(catalog, dict(spec["inventory"]), Surface(float(width), float(height)))
        suite.append((spec["name"], problem))
    return suite
