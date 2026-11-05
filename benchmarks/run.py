"""Run the benchmark: every method on every test instance, several seeds, equal wall-clock budget.

    uv run python -m benchmarks.run --budget 10 --seeds 5 --policy outputs/rl/policy.zip

Every returned loop is exported to JSON and re-validated by the full validator in the parent
process (the validity rate must be 100 %). Results go to ``outputs/benchmark`` (ignored by Git)
and, unless disabled, to a local MLflow store (``sqlite:///mlflow.db``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = ROOT / "benchmarks" / "instances"

_CTX = None
_PROBLEMS: dict = {}


def _init_worker(suite_path: str, policy_path: str | None) -> None:
    from benchmarks.methods import Context
    from slotor.envs import load_suite
    from slotor.service import default_catalog

    global _CTX, _PROBLEMS
    catalog = default_catalog()
    _PROBLEMS = dict(load_suite(suite_path, catalog, split="test"))
    _CTX = Context(catalog, policy_path)


def _run_task(task: tuple[str, str, int, float]) -> dict:
    from benchmarks.methods import METHODS
    from slotor.report import solution_to_json

    method, instance, seed, budget_s = task
    problem = _PROBLEMS[instance]
    started = time.monotonic()
    row = {"method": method, "instance": instance, "seed": seed, "budget_s": budget_s}
    try:
        output = METHODS[method](_CTX, problem, seed, budget_s)
    except Exception as exc:  # recorded, never hidden: a crash counts as no solution
        row.update(error=f"{type(exc).__name__}: {exc}", solution_json=None, history=[])
    else:
        row.update(
            error=None,
            history=[(float(t), float(length)) for t, length in output.history],
            solution_json=(
                solution_to_json(output.solution, problem) if output.solution is not None else None
            ),
        )
    row["elapsed_s"] = time.monotonic() - started
    return row


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "git_sha": run("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(run("status", "--porcelain")),
    }


def verify(row: dict, problems: dict) -> dict:
    """Independent verdict: re-validate the exported loop with the full validator."""
    from slotor.report import solution_from_json

    row = dict(row)
    row.update(found=row["solution_json"] is not None, valid=False, length_mm=0.0, n_pieces=0)
    row["first_valid_s"] = row["history"][0][0] if row["history"] else None
    if row["found"]:
        try:
            solution = solution_from_json(row["solution_json"], problems[row["instance"]])
        except ValueError as exc:
            row["error"] = (row["error"] or "") + f" INVALID EXPORT: {exc}"
        else:
            row.update(valid=True, length_mm=solution.length_mm, n_pieces=solution.n_pieces)
    return row


def log_mlflow(rows: list[dict], meta: dict, tracking_uri: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("slotor-benchmark")
    for row in rows:
        with mlflow.start_run(run_name=f"{row['method']}-{row['instance']}-s{row['seed']}"):
            mlflow.log_params(
                {
                    "method": row["method"],
                    "instance": row["instance"],
                    "seed": row["seed"],
                    "budget_s": row["budget_s"],
                    "catalog_sha256": meta["catalog_sha256"],
                    "git_sha": meta["git_sha"],
                    "git_dirty": meta["git_dirty"],
                    "tolerances": json.dumps(meta["tolerances"]),
                    "suite": meta["suite"],
                }
            )
            mlflow.log_metrics(
                {
                    "length_mm": row["length_mm"],
                    "valid": float(row["valid"]),
                    "n_pieces": row["n_pieces"],
                    "elapsed_s": row["elapsed_s"],
                }
            )
            if row["first_valid_s"] is not None:
                mlflow.log_metric("first_valid_s", row["first_valid_s"])
            for step, (_, length) in enumerate(row["history"]):
                mlflow.log_metric("length_mm_improvement", length, step=step)
            if row["solution_json"]:
                mlflow.log_text(row["solution_json"], "best.json")


def main(argv: list[str] | None = None) -> None:
    from benchmarks.methods import METHODS, NEEDS_POLICY
    from slotor.envs import load_suite
    from slotor.geometry import ToleranceConfig
    from slotor.service import default_catalog

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="default")
    parser.add_argument("--budget", type=float, default=10.0, help="wall-clock seconds per run")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--policy", default=None, help="trained policy (enables rl, beam_prior)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "benchmark"))
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    args = parser.parse_args(argv)
    if args.seeds < 5:
        parser.error("the protocol (D12) requires at least 5 seeds")

    suite_path = Path(args.suite) if Path(args.suite).exists() else SUITES / f"{args.suite}.json"
    catalog = default_catalog()
    problems = dict(load_suite(suite_path, catalog, split="test"))
    methods = args.methods or [m for m in METHODS if args.policy or m not in NEEDS_POLICY]
    if any(m in NEEDS_POLICY for m in methods) and not args.policy:
        parser.error("rl and beam_prior need --policy")
    tasks = [
        (method, instance, seed, args.budget)
        for seed in range(args.seeds)
        for instance in problems
        for method in methods
    ]
    print(f"{len(tasks)} runs, {args.workers} workers, budget {args.budget:.0f} s", flush=True)

    rows: list[dict] = []
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(str(suite_path), args.policy),
    ) as pool:
        futures = [pool.submit(_run_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(verify(future.result(), problems))
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({time.monotonic() - started:.0f} s)", flush=True)
    rows.sort(key=lambda r: (r["instance"], r["method"], r["seed"]))

    meta = {
        "suite": suite_path.stem,
        "budget_s": args.budget,
        "seeds": args.seeds,
        "methods": methods,
        "workers": args.workers,
        "catalog_sha256": catalog.content_sha256,
        "tolerances": asdict(ToleranceConfig()),
        "policy": args.policy,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "wall_s": time.monotonic() - started,
        **git_state(),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{suite_path.stem}-{args.budget:.0f}s.json"
    target.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1), encoding="utf-8")
    invalid = [r for r in rows if r["found"] and not r["valid"]]
    print(f"saved {target}; invalid exports: {len(invalid)}")
    if not args.no_mlflow:
        log_mlflow(rows, meta, args.tracking_uri)
        print(f"logged {len(rows)} runs to {args.tracking_uri}")
    if invalid:
        raise SystemExit("some methods returned invalid loops: this is a bug, see the results file")


if __name__ == "__main__":
    main()
