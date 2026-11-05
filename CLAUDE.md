# Slotor — instructions pour Claude

Optimiseur de circuits **Carrera Digital 132** : à partir d'un inventaire de rails et d'une surface rectangulaire,
trouver la boucle fermée **la plus longue** physiquement réalisable (à plat d'abord, puis avec ponts).
Algorithmes : seeds fermés → Beam Search → LNS ; R&D Reinforcement Learning (MaskablePPO) comparée à ces baselines.

Documents de référence (à lire avant de travailler) :
- [docs/MVP.md](docs/MVP.md) — vision initiale (certains choix sont remplacés, voir DECISIONS).
- [docs/PLAN.md](docs/PLAN.md) — phases 0 à 8 avec critères de sortie. **Travailler phase par phase.**
- [docs/DECISIONS.md](docs/DECISIONS.md) — décisions D1…D21. Ne pas les contredire sans ajouter une nouvelle entrée.
- [docs/SCHEMA.md](docs/SCHEMA.md) — base SQLite (DDL : `src/slotor/db/migrations/0001_init.sql`).
- [docs/OUTILS_EXISTANTS.md](docs/OUTILS_EXISTANTS.md) — analyse d'outils tiers (AutoGen) et idées retenues.

## État du projet

Phases 0 à 5 et 7 faites (RL + benchmark, D28 : le RL fait moins bien que la LNS, voir `benchmarks/REPORT.md`) ; phase 6 (ponts) : modèle et validateur faits, D30 (3 éléments pour monter, ≥ 1 de tablier, 3 pour descendre) ; graines à pont D31 (formes en S) ; phase 8 faite (D32, `slotor.purchase`, onglet « Que racheter ? ») (optimiseurs `src/slotor/optimizers` D25-D26 ; service `slotor.service` + UI Gradio `slotor.ui` D27) : moteur géométrique `src/slotor/geometry` (D23), validateur unique `src/slotor/validation` complet et
incrémental (D24), circuits d'or golden_a (valide) et golden_b (rejeté : croisement 2D). Phase 0 : `pyproject.toml`, migrations SQLite (`src/slotor/db`), seed JSON, chargeur de catalogue immuable et
import d'inventaire Track Architect (`src/slotor/catalog`) existent, avec leurs tests. `python -m slotor.ui` et `python -m benchmarks.run` existent ; vérifier qu'une commande
existe avant de l'annoncer comme fonctionnelle.

```bash
uv sync                      # environnement
uv run pytest                # tests (pytest + hypothesis)
uv run ruff check . && uv run ruff format --check .
uv run python -m slotor.ui   # interface Gradio locale
uv sync --all-extras         # + pile RL (torch, sb3-contrib, mlflow)
uv run python -m benchmarks.train_rl && uv run python -m benchmarks.run --policy outputs/rl/policy.zip --seeds 10
uv run python -m benchmarks.report   # écrit benchmarks/REPORT.md
```

## Disposition prévue

```
src/slotor/{db,catalog,geometry,validation,optimizers,envs,ui}/   # db/migrations/*.sql, catalog/seed/*.seed.json
benchmarks/   tests/   tests/fixtures/   data/   docs/   outputs/ (ignoré)
```
`medias/` = références visuelles locales, **ignorées par Git** (voir « Dépôt public »).

## Invariants — ne jamais les casser

1. **Un seul validateur** (D6). Beam, LNS, RL, UI et tests l'appellent ; ne jamais le réimplémenter ni le contourner.
   Un optimiseur ne marque aucun point pour un circuit invalide, mal fermé ou en chevauchement.
2. **Pose `(x, y, θ, level)`, `level` entier de 0 à 3** (D4 révisée par D30) : 2D à plat, aucune pente, **un seul étage** (niveau 3).
   Un pont = 3 éléments pour monter (niveaux 0→1→2→3), ≥ 1 de tablier, 3 pour descendre ; un élément de rampe n'a rien dessous ni
   dessus ; seul un tablier (niveau 3) peut recouvrir une pièce au sol. « À plat » = aucune pièce de niveau > 0.
3. **Catalogue = source de vérité** (D5, D20 : toute courbe a `radius_mm` + `radius_ref`) : chaque cote a une source. Valeur inconnue = `null` + refus, jamais une
   approximation tirée de la mémoire. Cotes de travail (D21, corrigées par D22) : **issues de la géométrie de Track Architect, pas mesurées sur le matériel** (`verified=true` = « adopté ») : rail 198 mm ;
   rayons de bord **extérieur** (`outer_edge`, D22, confirmé sur les emprises Track Architect) R1 396, R2 594, R3 792, R4 990 mm ; droites 345, 115 (1/3) et 86,25 (1/4) mm.
   Les circuits d'or de la phase 1 les contrôlent ; les bordures (Randstreifen) ne comptent pas dans l'emprise (D20). Seed : `src/slotor/catalog/seed/carrera_d132.seed.json`.
   **Ne pas moissonner Track Architect** (D19) : pas d'appels en masse à son serveur, pas de décodage de son obfuscation.
4. **Persistance SQLite** (D18) : la base ne sert pas aux boucles de calcul ; le catalogue est chargé en objets
   immuables. `solution.validated=1` n'est posé que par le validateur **complet**. Modifier le schéma = nouvelle migration
   numérotée, jamais une édition de `0001_init.sql` une fois appliqué.
5. **Unités** : mm et radians en interne, `float64` ; tolérances via `ToleranceConfig` (D3, D8, D9). Pas de nombre magique.
6. **Collision** (D8, D24) : aire d'intersection > ε **après érosion** des contours (pénétration > 2 mm), pas `overlaps()`/`intersects()`
   seuls ; STRtree en pré-filtre. Tolérances de fermeture (10 mm, 1 mm d'angle au bord) : expérience de l'utilisateur, à revoir après la feuille E1.
7. **Jamais de conclusion « le RL est meilleur »** sans le protocole du benchmark (D12) : budget égal, ≥ 5 seeds,
   validité 100 %, baselines beam et LNS.
8. L'UI Gradio ne contient **aucune logique métier** (D2).

## Conventions

- Docs en français ; code, identifiants, commentaires, messages d'erreur en anglais (D14).
- Python ≥ 3.11, type hints, `dataclass(frozen=True)`/pydantic pour les données du catalogue.
- Tests d'abord pour la géométrie et le validateur : tests négatifs (chevauchement, hors surface, inventaire dépassé,
  boucle non fermée, quasi-contact), property-based avec hypothesis, circuits d'or en JSON (D13).
- Reproductibilité : toute fonction stochastique prend un `seed`/`rng` explicite.
- Entraînement RL sur **CPU** (D11). Mesurer avant d'optimiser ; profiler le validateur incrémental.

## Dépôt public (D17)

Le dépôt GitHub est **public**. Donc :
- **`docs/` est privé** (ignoré par Git, D17) : il peut être absent sur un clone. Ne jamais retirer `/docs/` du `.gitignore`.
- Ne jamais committer `medias/` : les `IMG_*` contiennent des coordonnées GPS, les captures/visuels sont des images
  de tiers. Fixtures publiques = JSON reconstruits dans `tests/fixtures/`, ou images **sans EXIF** et libres de droits.
- Aucun secret, e-mail personnel, chemin absolu local ni donnée perso dans le code, les docs ou les logs.
- Avant tout commit : `git status` + relire le diff. Ne pas committer ni pousser sans demande explicite.
- Projet indépendant, **non affilié à Carrera** (marque citée à titre descriptif).

## Skills du projet (`.claude/skills/`)

| Skill | Quand l'utiliser |
|---|---|
| `track-geometry-testing` | Poses, connecteurs, fermeture, collisions Shapely, ponts, tests d'or/property-based |
| `combinatorial-search-lns` | Beam Search, LNS, seeds fermés, bornes d'élagage, mode anytime |
| `rl-env-design` | Environnement Gymnasium de construction de circuit : observations, actions, masques, récompense |
| `rl-maskable-ppo-training` | Entraînement MaskablePPO, curriculum, diagnostic des échecs |
| `rl-benchmark-mlflow` | Protocole de comparaison RL vs baselines, suivi MLflow, statistiques |

## Pièges connus

- Les croix jaunes des captures sont des **pièces de changement de voie** (double lane change), pas des croisements.
  Les croisements de `track-3/4/5` sont dessinés en 2D par Track Architect (aucune hauteur) ; en vrai ils exigent un
  **pont** (rampes, tablier, piliers) dont les pièces ne sont pas dans son catalogue.
- Un contact par arête n'est pas un chevauchement, même entre rails non adjacents (golden_a en est plein) ; seule une
  pénétration plus profonde que `collision_depth_mm` l'est (D24).
- Un masque d'actions entièrement faux bloque l'environnement RL : prévoir l'état « impasse » (voir `rl-env-design`).
- Pas de hauteur ni de pente dans le modèle (D4) : ne pas les introduire. Les pièces de pont sont les droites et courbes
  standard surélevées (D29, D30, réponses de l'utilisateur) : variantes dérivées du catalogue (`slotor.catalog.bridges`), même
  inventaire, escalier 3 + tablier + 3 ; seules les droites de 34,5 cm et les courbes sont levables (hypothèse à vérifier).
  Aucune cote de pont à inventer.
