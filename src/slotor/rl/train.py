"""MaskablePPO training on the construction environment (CPU, D11)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slotor.catalog.model import Catalog


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 500_000
    seed: int = 0
    n_envs: int = 4
    n_steps: int = 1024  # per env
    batch_size: int = 256
    learning_rate: float = 3e-4
    ent_coef: float = 0.02
    gamma: float = 1.0
    partial_start_prob: float = 0.5
    length_shaping: float = 0.5
    reward_reference: str = "stock"
    net_arch: tuple[int, ...] = (128, 128)


def train(catalog: Catalog, config: TrainConfig, out_path: str | Path, *, verbose: int = 0):
    """Train and save a policy; returns the model. Seeds python, numpy, torch and the envs."""
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from slotor.envs import InstanceSampler, TrackEnv

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)

    def make(rank: int):
        def factory():
            env = TrackEnv(
                catalog,
                sampler=InstanceSampler(catalog),
                gamma=config.gamma,
                length_shaping=config.length_shaping,
                reward_reference=config.reward_reference,
                partial_start_prob=config.partial_start_prob,
            )
            env.reset(seed=config.seed * 1000 + rank)
            return ActionMasker(env, lambda e: e.action_masks())

        return factory

    factories = [make(rank) for rank in range(config.n_envs)]
    vec = (
        SubprocVecEnv(factories, start_method="spawn")
        if config.n_envs > 1
        else DummyVecEnv(factories)
    )
    model = MaskablePPO(
        "MlpPolicy",
        vec,
        seed=config.seed,
        device="cpu",
        verbose=verbose,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        ent_coef=config.ent_coef,
        gamma=config.gamma,
        policy_kwargs={"net_arch": list(config.net_arch)},
    )
    model.learn(config.total_timesteps)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    vec.close()
    return model
