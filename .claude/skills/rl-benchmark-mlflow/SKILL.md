---
name: rl-benchmark-mlflow
description: Use when comparing Slotor optimizers (random, greedy, beam, LNS, RL) or tracking experiments with MLflow — fixed benchmark suite, equal wall-clock budgets, multi-seed statistics, reproducibility metadata, and honest reporting.
---

# Benchmark et suivi d'expériences

## Protocole (D12)
- **Suite d'instances figée** dans `benchmarks/instances/*.json` (surface, inventaire, marge, catalogue + hash). Inclure :
  instances petites (solution quasi connue), moyennes, et une réplique des inventaires lisibles sur les captures.
  Séparer instances d'**entraînement** et de **test** pour le RL (jamais d'évaluation sur l'entraînement).
- **Budget identique** en temps mur par méthode (10 s / 1 min / 5 min), même machine, même nombre de cœurs. Le temps
  d'entraînement RL est rapporté à part (coût amorti) ; l'inférence compte dans le budget.
- **≥ 5 seeds** (idéalement 10) par couple méthode × instance. Résumer par médiane + IQR et intervalle bootstrap.
- Comparer avec des tests appariés par instance (par ex. Wilcoxon) ou des intervalles bootstrap de différence ; corriger
  si beaucoup de comparaisons. Ne jamais publier la meilleure seed seule.

## Métriques
| Métrique | Pourquoi |
|---|---|
| Longueur médiane finale (mm) | objectif (D7) |
| Taux de validité | **doit être 100 %** (validateur complet) ; sinon bug |
| Longueur par rapport au meilleur connu | normalisation entre instances |
| Temps jusqu'à la première solution valide | « anytime » |
| Courbe longueur(t) | efficacité de la recherche |
| Nombre de pièces, longueur des deux voies | rapport utilisateur |

## Baselines (obligatoires)
aléatoire valide · glouton · beam · LNS. Une revendication RL exige : gain ≥ LNS à budget égal, **ou** accélération
mesurable vers une qualité donnée (temps pour atteindre X % du meilleur connu).

## MLflow (local)
```python
import mlflow

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)  # le backend fichier est déprécié dans les versions récentes ; vérifier
mlflow.set_experiment("slotor-benchmark")
with mlflow.start_run(run_name=f"{method}-{instance}-s{seed}"):
    mlflow.log_params(
        {
            "method": method,
            "seed": seed,
            "budget_s": budget,
            "catalog_hash": cat_hash,
            "git_sha": sha,
            "tolerances": tol_json,
        }
    )
    mlflow.log_metric("length_mm", length)  # + step=t pour la courbe
    mlflow.log_artifact("outputs/best.json")  # circuit + SVG
```
- Un run = une exécution (méthode × instance × seed). Hash du catalogue et SHA git **toujours** loggés ; refuser de
  tracer si l'arbre git est sale (ou marquer `dirty=true`).
- `mlflow.db`, `mlruns/` et `outputs/` sont ignorés par Git (dépôt public, D17) ; ne jamais y mettre de données personnelles.
- Rapport : `benchmarks/REPORT.md` généré par script (tableau + graphiques), reproductible via une seule commande.

## Honnêteté du rapport
- Dire si le RL perd. Lister hypothèses, budgets, limites (taille de la suite, variance).
- Séparer « meilleure solution trouvée » de « optimale prouvée » : l'heuristique ne prouve rien.
- Contrôler visuellement les 3 meilleurs circuits de chaque méthode (SVG à l'échelle) : un circuit « long » mais
  physiquement douteux invalide le résultat.
