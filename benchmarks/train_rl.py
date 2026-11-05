"""Train the MaskablePPO policy (formulation a) and record how it was trained.

uv run python -m benchmarks.train_rl --steps 500000 --out outputs/rl/policy.zip
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from slotor.rl.train import TrainConfig, train
from slotor.service import default_catalog


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--partial", type=float, default=0.5)
    parser.add_argument("--length-shaping", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--reward-reference", choices=["stock", "seed"], default="stock")
    parser.add_argument("--out", default="outputs/rl/policy.zip")
    args = parser.parse_args(argv)

    catalog = default_catalog()
    config = replace(
        TrainConfig(),
        total_timesteps=args.steps,
        seed=args.seed,
        n_envs=args.envs,
        partial_start_prob=args.partial,
        length_shaping=args.length_shaping,
        ent_coef=args.ent_coef,
        reward_reference=args.reward_reference,
    )
    started = time.monotonic()
    train(catalog, config, args.out, verbose=0)
    meta = {
        "config": asdict(config),
        "train_wall_s": time.monotonic() - started,
        "catalog_sha256": catalog.content_sha256,
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"trained in {meta['train_wall_s']:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
