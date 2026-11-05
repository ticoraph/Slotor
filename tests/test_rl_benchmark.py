"""RL inference, benchmark methods, statistics, runner and report (phase 7)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from benchmarks import methods, report, run, stats  # noqa: E402

from slotor.optimizers import BeamConfig, beam_search  # noqa: E402
from slotor.report import solution_to_json  # noqa: E402
from slotor.rl.agent import PolicyAgent  # noqa: E402
from slotor.rl.train import TrainConfig, train  # noqa: E402
from slotor.validation import validate_layout  # noqa: E402

from .conftest import FIXTURES, cached_catalog  # noqa: E402
from .test_optimizers import problem  # noqa: E402


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("rl") / "policy.zip"
    config = TrainConfig(total_timesteps=2048, n_envs=1, n_steps=256, batch_size=64)
    train(cached_catalog(), config, path)
    return path


@pytest.fixture(scope="module")
def agent(tiny_policy) -> PolicyAgent:
    return PolicyAgent(tiny_policy, cached_catalog())


def small_problem():
    from slotor.optimizers import Problem
    from slotor.validation import Surface

    inventory = {"curve_r1_60": 6, "straight_345": 2, "straight_115": 3}
    return Problem(cached_catalog(), inventory, Surface(1400, 1200))


def assert_valid(prob, solution) -> None:
    result = validate_layout(
        solution.items, prob.inventory, surface=prob.surface, start=solution.start, tol=prob.tol
    )
    assert result.valid, result.issues


# --- policy inference ----------------------------------------------------------------------------


def test_policy_probabilities_cover_exactly_the_valid_actions(agent):
    from slotor.envs import ConstructionState

    state = ConstructionState.initial(problem("kit_mixed"), agent.table)
    probs = agent.probabilities(state)
    mask = state.action_mask()
    assert probs.sum() == pytest.approx(1.0)
    assert (probs[~mask] == 0).all() and (probs[mask] > 0).all()


def test_rollouts_return_validated_loops_or_nothing(agent):
    prob = problem("kit_oval")
    best, history = agent.best_of_rollouts(prob, random.Random(0), budget_s=1.0)
    if best is not None:
        assert_valid(prob, best)
        assert history[-1][1] == pytest.approx(best.length_mm)


def test_a_policy_for_another_catalog_is_refused(tiny_policy, monkeypatch):
    catalog = cached_catalog()
    import dataclasses

    shrunk = dataclasses.replace(catalog, pieces=catalog.pieces[:3])
    with pytest.raises(ValueError, match="action"):
        PolicyAgent(tiny_policy, shrunk)


def test_beam_with_a_learned_prior_stays_valid_and_deterministic(agent):
    prob = problem("kit_mixed")
    config = BeamConfig(
        width=40,
        max_expansions=3000,
        prior=agent.log_prior(prob),
        prior_weight=methods.PRIOR_WEIGHT_MM,
    )
    first = beam_search(prob, seed=4, config=config)
    second = beam_search(prob, seed=4, config=config)
    assert first.best is not None
    assert_valid(prob, first.best)
    assert first.best.slugs == second.best.slugs


def test_a_prior_only_reranks_it_never_prunes(agent):
    """The same bounds apply: the set of reachable closed loops is unchanged by the prior."""
    prob = problem("kit_oval")
    plain = beam_search(prob, seed=0, config=BeamConfig(width=400, use_seeds=False))
    biased = beam_search(
        prob,
        seed=0,
        config=BeamConfig(
            width=400, use_seeds=False, prior=agent.log_prior(prob), prior_weight=500.0
        ),
    )
    assert plain.best is not None and biased.best is not None
    assert biased.best.length_mm == pytest.approx(plain.best.length_mm)


# --- benchmark methods ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["random", "greedy", "beam", "lns", "pipeline"])
def test_each_baseline_returns_a_valid_loop_or_nothing(name):
    ctx = methods.Context(cached_catalog(), None)
    prob = small_problem()
    output = methods.METHODS[name](ctx, prob, 0, 1.0)
    if output.solution is not None:
        assert_valid(prob, output.solution)
        assert output.history and output.history[-1][1] == pytest.approx(output.solution.length_mm)


def test_policy_methods_run_with_a_policy(tiny_policy):
    ctx = methods.Context(cached_catalog(), tiny_policy)
    prob = small_problem()
    for name in ("rl", "beam_prior"):
        output = methods.METHODS[name](ctx, prob, 1, 1.0)
        if output.solution is not None:
            assert_valid(prob, output.solution)


def test_policy_methods_without_a_policy_say_so():
    with pytest.raises(RuntimeError, match="trained policy"):
        methods.Context(cached_catalog(), None).agent  # noqa: B018


def test_random_valid_is_seeded():
    ctx = methods.Context(cached_catalog(), None)
    prob = problem("kit_oval")
    a = methods.random_valid(ctx, prob, 3, 0.5)
    b = methods.random_valid(ctx, prob, 3, 0.5)
    assert (a.solution is None) == (b.solution is None)


# --- statistics ----------------------------------------------------------------------------------


def make_rows(values: dict[tuple[str, str], list[float]]) -> list[dict]:
    rows = []
    for (method, instance), lengths in values.items():
        for seed, length in enumerate(lengths):
            rows.append(
                {
                    "method": method,
                    "instance": instance,
                    "seed": seed,
                    "valid": length > 0,
                    "found": length > 0,
                    "length_mm": float(length),
                    "history": [(0.1, float(length))] if length > 0 else [],
                    "first_valid_s": 0.1 if length > 0 else None,
                    "elapsed_s": 1.0,
                    "budget_s": 1.0,
                }
            )
    return rows


def test_bootstrap_interval_brackets_the_median():
    values = [10, 11, 12, 13, 14, 15, 16]
    low, high = stats.bootstrap_ci(values)
    assert low <= np.median(values) <= high


def test_normalization_uses_the_best_known_length_and_zero_for_failures():
    rows = make_rows({("a", "i"): [100, 80], ("b", "i"): [50, 0]})
    scores = {(r["method"], r["seed"]): r["score"] for r in stats.normalized(rows)}
    assert scores[("a", 0)] == 1.0 and scores[("a", 1)] == 0.8
    assert scores[("b", 0)] == 0.5 and scores[("b", 1)] == 0.0


def test_paired_difference_detects_a_clear_gap_and_no_gap():
    better = make_rows(
        {(m, i): [lengths] * 5 for m, lengths in (("a", 100), ("b", 60)) for i in ("x", "y", "z")}
    )
    diff, (low, high), _ = stats.paired_difference(better, "a", "b")
    assert diff == pytest.approx(0.4) and low > 0
    same = make_rows({(m, i): [100, 90, 95, 100, 92] for m in ("a", "b") for i in ("x", "y", "z")})
    diff, (low, high), _ = stats.paired_difference(same, "a", "b")
    assert diff == pytest.approx(0.0) and low <= 0 <= high


# --- runner and report -------------------------------------------------------------------------


def test_verify_marks_a_tampered_export_invalid():
    prob = problem("kit_oval")
    from slotor.optimizers import generate_seeds

    solution = generate_seeds(prob)[0]
    good = json.loads(solution_to_json(solution, prob))
    row = {
        "solution_json": json.dumps(good),
        "history": [(0.1, solution.length_mm)],
        "error": None,
        "instance": "kit_oval",
    }
    assert run.verify(row, {"kit_oval": prob})["valid"]
    good["pieces"].pop()
    bad = run.verify({**row, "solution_json": json.dumps(good)}, {"kit_oval": prob})
    assert bad["found"] and not bad["valid"] and "INVALID EXPORT" in bad["error"]
    none = run.verify({**row, "solution_json": None, "history": []}, {"kit_oval": prob})
    assert not none["found"] and not none["valid"] and none["length_mm"] == 0.0


def test_runner_end_to_end_with_mlflow_and_report(tmp_path, monkeypatch):
    out = tmp_path / "bench"
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run.main(
        [
            "--suite", str(FIXTURES / "bench_tiny.json"),
            "--budget", "1", "--seeds", "5", "--workers", "2",
            "--methods", "random", "greedy", "lns",
            "--out", str(out), "--tracking-uri", uri,
        ]
    )  # fmt: skip
    results = json.loads((out / "bench_tiny-1s.json").read_text(encoding="utf-8"))
    rows = results["rows"]
    assert len(rows) == 2 * 3 * 5
    assert all(r["valid"] or not r["found"] for r in rows)  # invalid exports would have raised
    assert results["meta"]["catalog_sha256"] == cached_catalog().content_sha256
    assert "git_sha" in results["meta"] and "tolerances" in results["meta"]
    text = report.build_report(results, None)
    assert "Verdict" in text and "greedy" in text and "Circuits invalides : **0**" in text
    import mlflow

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["slotor-benchmark"])
    assert len(runs) == len(rows)
    assert {"params.catalog_sha256", "params.git_sha", "metrics.length_mm"} <= set(runs.columns)


def test_runner_refuses_fewer_than_five_seeds(tmp_path):
    with pytest.raises(SystemExit):
        run.main(["--suite", str(FIXTURES / "bench_tiny.json"), "--seeds", "3", "--no-mlflow"])


def test_the_frozen_suite_is_test_only_and_valid():
    from slotor.envs import load_suite

    suite = load_suite(Path("benchmarks/instances/default.json"), cached_catalog(), split="test")
    assert len(suite) >= 8
    names = [n for n, _ in suite]
    assert len(set(names)) == len(names) and {"golden_a", "golden_b"} <= set(names)
