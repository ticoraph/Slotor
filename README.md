# Slotor

Given an inventory of rails and a rectangular surface, find the **longest closed loop** that can physically be built
with Carrera Digital 132 slot-car track (flat first, then with bridges).

Planned approach: closed seeds, then Beam Search, then Large Neighborhood Search (LNS). A Reinforcement Learning
study (MaskablePPO) is compared against these baselines under an equal-budget protocol.

> Independent project, **not affiliated with Carrera**. The brand name is used for descriptive purposes only.

## Status

Phase 0 (SQLite schema, JSON catalog seed, immutable catalog loader, Track Architect inventory import) is done.
Phase 1 (geometry engine), phase 2 (single validator, full and incremental), phase 3 (closed seeds,
Beam Search), phase 4 (LNS), phase 5 (local Gradio UI), phase 6 (bridges: 3 elements up, a deck, 3 down; validator done; bridged seeds from figure-S loops), phase 8 (which packs to buy) and phase 7 (RL study and benchmark) are done.
In that study the learned policy does *not* beat the LNS at equal budget: see `benchmarks/REPORT.md`.

## Development

```bash
uv sync                                  # environment (Python >= 3.11)
uv sync --all-extras                     # + RL stack (torch, sb3-contrib, mlflow) for phase 7
uv run pytest                            # tests (pytest + hypothesis)
uv run ruff check . && uv run ruff format --check .
uv run python -m slotor.ui               # local web interface (127.0.0.1 only)
```

All catalog dimensions come from documented sources with provenance, never from memory. Unmeasured working values
are flagged in the seed (`src/slotor/catalog/seed/carrera_d132.seed.json`).
