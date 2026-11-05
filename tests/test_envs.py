"""The construction environment (formulation a): invariants required before any training."""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from slotor.envs import ActionTable, ConstructionState, InstanceSampler, TrackEnv  # noqa: E402
from slotor.envs.track_env import DEADLOCK_PENALTY, INVALID_ACTION_PENALTY  # noqa: E402
from slotor.optimizers import Problem  # noqa: E402
from slotor.validation import Surface, validate_layout  # noqa: E402

from .conftest import cached_catalog  # noqa: E402
from .test_optimizers import problem  # noqa: E402


def sampled_env(**kwargs) -> TrackEnv:
    catalog = cached_catalog()
    return TrackEnv(catalog, sampler=InstanceSampler(catalog), **kwargs)


def play_random(env: TrackEnv, rng: np.random.Generator, seed: int):
    """One masked-random episode: returns (total reward, final info, step count)."""
    env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        mask = env.action_masks()
        assert mask.any(), "an agent must never face an all-false mask"
        _, reward, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(mask))))
        total += reward
        steps += 1
        if terminated or truncated:
            return total, info, steps


def test_gymnasium_checker_passes():
    check_env(sampled_env(), skip_render_check=True)


def test_action_table_is_independent_of_the_inventory_and_versioned():
    catalog = cached_catalog()
    table = ActionTable.from_catalog(catalog)
    assert table.catalog_sha256 == catalog.content_sha256
    assert ("curve_r1_60", True) in table.entries and ("straight_345", True) not in table.entries
    assert len(table) == 13
    env = sampled_env()
    env.reset(seed=1)
    first = env.table
    env.reset(seed=2)
    assert env.table is first and env.action_space.n == len(table)


def test_observation_is_finite_bounded_and_fixed_size():
    env = sampled_env()
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    while True:
        assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
        assert env.observation_space.contains(obs)
        obs, _, terminated, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        if terminated:
            break


def test_same_seed_same_trajectory():
    def trajectory(seed: int):
        env = sampled_env()
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        out = []
        while True:
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
            obs, reward, terminated, _, _ = env.step(action)
            out.append((action, round(reward, 9), obs.tobytes()))
            if terminated:
                return out

    assert trajectory(11) == trajectory(11)
    assert trajectory(11) != trajectory(12)


def test_return_of_an_episode_is_the_validated_length_or_a_penalty():
    """Potential shaping telescopes: a closed loop returns length / reference, a dead end -0.2."""
    env = sampled_env(length_shaping=0.5)
    rng = np.random.default_rng(0)
    closed = deadlocks = 0
    for seed in range(400):
        total, info, _ = play_random(env, rng, seed)
        if info.get("closed"):
            closed += 1
            assert total == pytest.approx(env.last_solution.length_mm / env._reference)
        else:
            deadlocks += 1
            assert info.get("deadlock")
            assert total == pytest.approx(DEADLOCK_PENALTY)
    assert closed + deadlocks == 400


def test_no_episode_is_rewarded_without_a_validated_closed_loop():
    env = sampled_env(partial_start_prob=0.5, length_shaping=1.0, shaping=1.0)
    rng = np.random.default_rng(1)
    for seed in range(300):
        total, info, _ = play_random(env, rng, seed)
        if not info.get("closed"):
            assert total <= 0.0


def test_ten_thousand_masked_episodes_never_close_an_invalid_loop():
    """Required invariant: every loop the environment calls closed passes the full validator."""
    env = sampled_env(partial_start_prob=0.7)
    rng = np.random.default_rng(2)
    closed = 0
    for seed in range(10_000):
        _, info, _ = play_random(env, rng, 100_000 + seed)
        if info.get("closed"):
            closed += 1
            solution = env.last_solution
            problem_ = env.state.problem
            result = validate_layout(
                solution.items,
                problem_.inventory,
                surface=problem_.surface,
                start=solution.start,
                tol=problem_.tol,
            )
            assert result.valid, result.issues
    assert closed > 500, "partial starts should close often enough for this test to mean something"


def test_a_masked_out_action_is_never_applied():
    env = sampled_env()
    env.reset(seed=5)
    before = env.state.n_pieces
    invalid = int(np.flatnonzero(~env.action_masks())[0])
    _, reward, terminated, _, info = env.step(invalid)
    assert terminated and info["invalid_action"] and reward == INVALID_ACTION_PENALTY
    assert env.state.n_pieces == before


def test_an_instance_with_no_feasible_start_still_gives_a_well_formed_mask():
    """The bound rules out every first piece: the mask stays well-formed and any step ends."""
    prob = Problem(cached_catalog(), {"straight_345": 1}, Surface(1000, 1000))
    env = TrackEnv(cached_catalog(), problems=[prob])
    env.reset(seed=0)
    assert env.action_masks().any()
    _, reward, terminated, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert terminated and info["invalid_action"] and reward == INVALID_ACTION_PENALTY


def test_the_oval_kit_closes_with_the_whole_inventory():
    prob = problem("kit_oval")
    state = ConstructionState.initial(prob, ActionTable.from_catalog(cached_catalog()))
    table = state.table
    sequence = [("curve_r1_60", False)] * 3 + [("straight_345", False)] * 2
    sequence += [("curve_r1_60", False)] * 3 + [("straight_345", False)] * 2
    for entry in sequence:
        state = state.children()[table.entries.index(entry)]
    assert state.closed
    assert state.solution().length_mm == pytest.approx(3246.106036232337)


def test_partial_start_gives_a_repair_task():
    env = sampled_env(partial_start_prob=1.0)
    prefixes = []
    for seed in range(40):
        env.reset(seed=seed)
        prefixes.append(env.state.n_pieces)
    assert max(prefixes) > 0
    assert all(not env.state.closed for _ in [0])


def test_environment_speed_is_reasonable():
    env = sampled_env()
    rng = np.random.default_rng(0)
    steps, started = 0, time.perf_counter()
    for seed in range(150):
        steps += play_random(env, rng, seed)[2]
    rate = steps / (time.perf_counter() - started)
    assert rate > 500, f"{rate:.0f} steps/s is too slow to train on"


def test_constructor_requires_exactly_one_instance_source():
    catalog = cached_catalog()
    with pytest.raises(ValueError):
        TrackEnv(catalog)
    with pytest.raises(ValueError):
        TrackEnv(catalog, problems=[problem("kit_oval")], sampler=InstanceSampler(catalog))
    with pytest.raises(ValueError):
        TrackEnv(catalog, sampler=InstanceSampler(catalog), reward_reference="moon")


def test_sampler_is_deterministic_and_documented_ranges():
    sampler = InstanceSampler(cached_catalog())
    a = sampler.sample(np.random.default_rng(4))
    b = sampler.sample(np.random.default_rng(4))
    assert dict(a.inventory) == dict(b.inventory) and a.surface == b.surface
    for seed in range(50):
        p = sampler.sample(np.random.default_rng(seed))
        assert 1500 <= p.surface.width_mm <= 3200 and 800 <= p.surface.height_mm <= 2000
        assert 4 <= p.inventory["curve_r1_60"] <= 14 and 2 <= p.inventory["straight_345"] <= 16
