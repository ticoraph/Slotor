"""Generate ``benchmarks/REPORT.md`` (and a figure) from a benchmark results file.

    uv run python -m benchmarks.report outputs/benchmark/default-10s.json

The verdict sentences are derived from the numbers by fixed rules, not written by hand.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.stats import (
    best_known,
    paired_difference,
    pooled_score,
    quantiles,
)

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["random", "greedy", "beam", "lns", "lns_bandit", "pipeline", "rl", "beam_prior"]
DESCRIPTION = {
    "random": "rollouts aléatoires masqués (aucune graine de départ)",
    "greedy": "pièce valide la plus longue, ε-glouton (aucune graine de départ)",
    "beam": "seed en stade puis Beam Search (largeur 200)",
    "lns": "seed en stade puis LNS",
    "lns_bandit": "LNS dont l'opérateur et la taille de fenêtre sont choisis par bandit de Thompson (adaptatif, sans apprentissage entre instances ; ce n'est pas la formulation c)",
    "pipeline": "seeds → beam (20 %) → LNS (service de l'UI)",
    "rl": "politique MaskablePPO apprise, rollouts échantillonnés (formulation a)",
    "beam_prior": "Beam Search avec la politique apprise comme prior de classement (formulation b)",
}


def _fmt_ci(point: float, ci: tuple[float, float]) -> str:
    return f"{point:.3f} [{ci[0]:.3f} ; {ci[1]:.3f}]"


def _verdict(name_a: str, name_b: str, diff: float, ci: tuple[float, float]) -> str:
    low, high = ci
    if low > 0:
        return f"**{name_a} fait mieux que {name_b}** (écart {diff:+.3f}, IC95 % [{low:+.3f} ; {high:+.3f}])."
    if high < 0:
        return f"**{name_a} fait moins bien que {name_b}** (écart {diff:+.3f}, IC95 % [{low:+.3f} ; {high:+.3f}])."
    return (
        f"Aucune différence démontrée entre {name_a} et {name_b} "
        f"(écart {diff:+.3f}, IC95 % [{low:+.3f} ; {high:+.3f}] contient 0)."
    )


def _time_to_quality(rows: list[dict], best: dict[str, float], fraction: float = 0.9):
    out: dict[str, list[float | None]] = defaultdict(list)
    for row in rows:
        target = fraction * best.get(row["instance"], 0.0)
        reached = next((t for t, length in row["history"] if length >= target and target > 0), None)
        out[row["method"]].append(reached)
    return out


def _figure(rows: list[dict], methods: list[str], budget: float, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    best = best_known(rows)
    grid = np.linspace(0, budget, 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        curves = []
        for row in rows:
            if row["method"] != method or best.get(row["instance"], 0) <= 0:
                continue
            curve = np.zeros_like(grid)
            for t, length in row["history"]:
                curve[grid >= t] = max(length, 0) / best[row["instance"]]
            curves.append(np.maximum.accumulate(curve))
        if curves:
            ax.plot(grid, np.median(curves, axis=0), label=method)
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("longueur / meilleur connu (médiane)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def build_report(data: dict, figure_name: str | None) -> str:
    meta, rows = data["meta"], data["rows"]
    methods = [m for m in ORDER if m in {r["method"] for r in rows}]
    instances = sorted({r["instance"] for r in rows})
    best = best_known(rows)
    invalid = [r for r in rows if r["found"] and not r["valid"]]
    overruns = [r for r in rows if r["elapsed_s"] > 1.15 * r["budget_s"] + 1.0]
    policy_meta = None
    if meta.get("policy"):
        meta_path = Path(meta["policy"]).with_suffix(".json")
        if meta_path.exists():
            policy_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    out = ["# Rapport de benchmark (phase 7, M3)", ""]
    out += [
        "Généré par `benchmarks/report.py` à partir d'un fichier de résultats ; ne pas modifier à la main.",
        "Reproduction : `uv run python -m benchmarks.train_rl` puis `uv run python -m benchmarks.run "
        f"--budget {meta['budget_s']:.0f} --seeds {meta['seeds']} --policy outputs/rl/policy.zip` "
        "puis `uv run python -m benchmarks.report`.",
        "",
        "## Protocole",
        "",
        f"- Budget : **{meta['budget_s']:.0f} s de temps mur par exécution**, identique pour toutes les méthodes ; "
        f"{meta['seeds']} graines par couple méthode × instance ; {len(instances)} instances de test figées "
        f"(`benchmarks/instances/{meta['suite']}.json`).",
        f"- Machine : {meta['platform']}, {meta['cpu_count']} cœurs, Python {meta['python']}, "
        f"{meta['workers']} exécutions en parallèle (un thread chacune). Le temps d'inférence du RL compte dans le budget ; "
        "son temps d'entraînement est rapporté à part.",
        f"- Code : `{meta['git_sha'][:10]}`{' (arbre modifié)' if meta['git_dirty'] else ''} ; "
        f"catalogue `{meta['catalog_sha256'][:12]}`.",
        "- Chaque circuit rendu est **exporté en JSON puis revalidé par le validateur complet** dans le processus parent. "
        f"Circuits invalides : **{len(invalid)}** (doit être 0). Exécutions qui dépassent le budget de plus de 15 % + 1 s : {len(overruns)}.",
        "- Score normalisé = longueur / meilleur circuit connu sur l'instance (toutes méthodes et graines confondues) ; "
        "0 si la méthode n'a trouvé aucun circuit valide.",
        "",
        "Méthodes :",
        "",
    ]
    out += [f"- `{m}` : {DESCRIPTION[m]}" for m in methods]
    if policy_meta:
        cfg = policy_meta["config"]
        out += [
            "",
            f"Politique : {cfg['total_timesteps']:,} pas d'entraînement, {cfg['n_envs']} environnements, graine {cfg['seed']}, "
            f"départ partiel {cfg['partial_start_prob']}, façonnage de longueur {cfg['length_shaping']}, "
            f"référence de récompense `{cfg.get('reward_reference', 'stock')}`, "
            f"{policy_meta['train_wall_s'] / 60:.1f} min sur CPU.",
        ]

    out += [
        "",
        "## Résultat agrégé",
        "",
        "| Méthode | Score normalisé médian moyen [IC95 %] | Circuit trouvé | Temps jusqu'au 1er circuit (médiane) |",
        "|---|---|---|---|",
    ]
    for method in methods:
        point, ci = pooled_score(rows, method)
        mrows = [r for r in rows if r["method"] == method]
        found = sum(r["valid"] for r in mrows) / len(mrows)
        firsts = [r["first_valid_s"] for r in mrows if r["first_valid_s"] is not None]
        first = f"{np.median(firsts):.2f} s" if firsts else "—"
        out.append(f"| {method} | {_fmt_ci(point, ci)} | {100 * found:.0f} % | {first} |")

    out += [
        "",
        "## Par instance",
        "",
        "Longueur médiane en mm (IQR) ; entre crochets : part des graines qui ont trouvé un circuit.",
        "",
    ]
    out += [
        "| Instance | Meilleur connu | " + " | ".join(methods) + " |",
        "|---|---|" + "---|" * len(methods),
    ]
    for instance in instances:
        cells = []
        for method in methods:
            mrows = [r for r in rows if r["instance"] == instance and r["method"] == method]
            lengths = [r["length_mm"] for r in mrows if r["valid"]]
            rate = len(lengths) / len(mrows) if mrows else 0
            if lengths:
                q1, med, q3 = quantiles(lengths)
                cells.append(f"{med:.0f} ({q1:.0f}–{q3:.0f}) [{100 * rate:.0f} %]")
            else:
                cells.append(f"— [{100 * rate:.0f} %]")
        out.append(f"| {instance} | {best.get(instance, 0):.0f} | " + " | ".join(cells) + " |")

    out += [
        "",
        "## Temps pour atteindre 90 % du meilleur connu",
        "",
        "| Méthode | Part des exécutions qui y arrivent | Temps médian parmi celles-ci |",
        "|---|---|---|",
    ]
    ttq = _time_to_quality(rows, best)
    for method in methods:
        times = ttq[method]
        reached = [t for t in times if t is not None]
        med = f"{np.median(reached):.2f} s" if reached else "—"
        out.append(f"| {method} | {100 * len(reached) / len(times):.0f} % | {med} |")

    if figure_name:
        out += ["", f"![longueur en fonction du temps]({figure_name})", ""]

    out += [
        "",
        "## Comparaisons appariées par instance",
        "",
        "Écart de score normalisé (A − B), moyenné sur les instances ; IC95 % par bootstrap sur les graines ; test de Wilcoxon sur les médianes par instance (peu de puissance avec si peu d'instances : l'intervalle prime).",
        "",
        "| A | B | Écart | IC95 % | p (Wilcoxon) |",
        "|---|---|---|---|---|",
    ]
    pairs = [
        ("rl", "lns"),
        ("rl", "random"),
        ("rl", "greedy"),
        ("beam_prior", "beam"),
        ("beam", "lns"),
        ("pipeline", "lns"),
        ("lns", "random"),
    ]
    verdicts = []
    for a, b in pairs:
        if a not in methods or b not in methods:
            continue
        diff, ci, p_value = paired_difference(rows, a, b)
        out.append(
            f"| {a} | {b} | {diff:+.3f} | [{ci[0]:+.3f} ; {ci[1]:+.3f}] | {'—' if p_value is None else f'{p_value:.3f}'} |"
        )
        verdicts.append((a, b, diff, ci))

    out += ["", "## Verdict (règles fixes sur les intervalles)", ""]
    for a, b, diff, ci in verdicts:
        if (a, b) in {
            ("rl", "lns"),
            ("beam_prior", "beam"),
            ("rl", "greedy"),
            ("pipeline", "lns"),
            ("lns_bandit", "lns"),
        }:
            out.append(f"- {_verdict(a, b, diff, ci)}")
    out += [
        "",
        "## Limites",
        "",
        f"- {len(instances)} instances et {meta['seeds']} graines : les intervalles sont larges, une différence modeste peut passer inaperçue.",
        "- Une seule politique entraînée (une graine d'entraînement) : la variance d'entraînement n'est pas mesurée.",
        "- `beam` et `lns` partent d'un seed en stade (D10) ; `random`, `greedy` et `rl` construisent depuis zéro : la comparaison inclut cet avantage de départ.",
        "- Seules les droites et les courbes sont modélisées ; les pièces spéciales des captures sont ignorées.",
        "- « Meilleur connu » n'est pas un optimum prouvé : c'est le meilleur circuit vu dans ce benchmark.",
        "- Contrôle visuel : voir les SVG exportés (`outputs/benchmark/svg/`) avant de se fier à un score.",
        "",
    ]
    return "\n".join(out)


def export_best_svgs(data: dict, directory: Path) -> int:
    """SVG of the best loop of each method on each instance, for the mandatory visual check."""
    from slotor.envs import load_suite
    from slotor.report import solution_from_json, solution_to_svg
    from slotor.service import default_catalog

    suite = Path(__file__).parent / "instances" / f"{data['meta']['suite']}.json"
    if not suite.exists():
        return 0
    problems = dict(load_suite(suite, default_catalog(), split="test"))
    best: dict[tuple[str, str], dict] = {}
    for row in data["rows"]:
        if row["valid"]:
            key = (row["instance"], row["method"])
            if key not in best or row["length_mm"] > best[key]["length_mm"]:
                best[key] = row
    directory.mkdir(parents=True, exist_ok=True)
    for (instance, method), row in best.items():
        problem = problems[instance]
        solution = solution_from_json(row["solution_json"], problem)
        (directory / f"{instance}-{method}.svg").write_text(
            solution_to_svg(solution, problem), encoding="utf-8"
        )
    return len(best)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    results = Path(args[0]) if args else ROOT / "outputs" / "benchmark" / "default-10s.json"
    data = json.loads(results.read_text(encoding="utf-8"))
    figure = ROOT / "benchmarks" / "length_over_time.png"
    methods = [m for m in ORDER if m in {r["method"] for r in data["rows"]}]
    has_figure = _figure(data["rows"], methods, data["meta"]["budget_s"], figure)
    report = build_report(data, figure.name if has_figure else None)
    (ROOT / "benchmarks" / "REPORT.md").write_text(report, encoding="utf-8")
    count = export_best_svgs(data, ROOT / "outputs" / "benchmark" / "svg")
    print(f"wrote benchmarks/REPORT.md ({count} SVG in outputs/benchmark/svg)")


if __name__ == "__main__":
    main()
