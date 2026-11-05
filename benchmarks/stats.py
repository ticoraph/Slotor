"""Statistics for the benchmark: medians, bootstrap intervals, paired comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

BOOTSTRAP_RESAMPLES = 4000
STAT_SEED = 20260511


def quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return float(q1), float(med), float(q3)


def bootstrap_ci(
    values: Sequence[float], statistic=np.median, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval of ``statistic`` over ``values``."""
    data = np.asarray(values, dtype=float)
    if len(data) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(STAT_SEED)
    draws = rng.choice(data, size=(BOOTSTRAP_RESAMPLES, len(data)), replace=True)
    stats = statistic(draws, axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def best_known(rows: Sequence[dict]) -> dict[str, float]:
    """Longest valid loop seen on each instance, over every method and seed of the benchmark."""
    best: dict[str, float] = defaultdict(float)
    for row in rows:
        if row["valid"]:
            best[row["instance"]] = max(best[row["instance"]], row["length_mm"])
    return dict(best)


def normalized(rows: Sequence[dict]) -> list[dict]:
    """Rows with ``score`` = length / best known (0 for a method that found no valid loop)."""
    best = best_known(rows)
    out = []
    for row in rows:
        reference = best.get(row["instance"], 0.0)
        score = row["length_mm"] / reference if row["valid"] and reference > 0 else 0.0
        out.append({**row, "score": score})
    return out


def method_scores(rows: Sequence[dict], method: str) -> dict[str, list[float]]:
    """Normalized scores of ``method`` grouped by instance (one value per seed)."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in normalized(rows):
        if row["method"] == method:
            grouped[row["instance"]].append(row["score"])
    return dict(grouped)


def pooled_score(rows: Sequence[dict], method: str) -> tuple[float, tuple[float, float]]:
    """Mean over instances of the per-instance median score, with a bootstrap interval.

    The bootstrap resamples seeds within each instance, so the interval reflects run-to-run
    variance (not the choice of instances).
    """
    per_instance = method_scores(rows, method)
    if not per_instance:
        return float("nan"), (float("nan"), float("nan"))
    point = float(np.mean([np.median(v) for v in per_instance.values()]))
    rng = np.random.default_rng(STAT_SEED)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draws.append(
            np.mean(
                [np.median(rng.choice(v, size=len(v), replace=True)) for v in per_instance.values()]
            )
        )
    return point, (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def paired_difference(
    rows: Sequence[dict], method_a: str, method_b: str
) -> tuple[float, tuple[float, float], float | None]:
    """``a - b`` on the pooled score: point estimate, bootstrap interval, Wilcoxon p-value.

    The p-value is a signed-rank test on the per-instance median differences; with only a handful
    of instances it has little power, so the interval is the primary evidence.
    """
    from scipy.stats import wilcoxon

    a, b = method_scores(rows, method_a), method_scores(rows, method_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("nan"), (float("nan"), float("nan")), None
    diffs = [float(np.median(a[i]) - np.median(b[i])) for i in shared]
    rng = np.random.default_rng(STAT_SEED)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        draws.append(
            np.mean(
                [
                    np.median(rng.choice(a[i], size=len(a[i])))
                    - np.median(rng.choice(b[i], size=len(b[i])))
                    for i in shared
                ]
            )
        )
    low, high = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    p_value = None
    if any(d != 0 for d in diffs):
        try:
            p_value = float(wilcoxon(diffs).pvalue)
        except ValueError:
            p_value = None
    return float(np.mean(diffs)), (low, high), p_value
