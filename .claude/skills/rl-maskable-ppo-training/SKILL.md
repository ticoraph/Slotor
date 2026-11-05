---
name: rl-maskable-ppo-training
description: Use when training or debugging a MaskablePPO agent (sb3-contrib) on Slotor's track-construction environment — setup, hyperparameter starting points, curriculum, parallel envs on CPU, evaluation with masks, and diagnosing failures such as entropy collapse or invalid-action deadlocks.
---

# Entraînement MaskablePPO

Vérifier les signatures dans la version installée de `sb3-contrib` / `stable-baselines3` (l'API évolue) avant de coder.

## Mise en place
```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy

env = ActionMasker(TrackEnv(cfg), lambda e: e.action_masks())
model = MaskablePPO("MlpPolicy", env, seed=seed, device="cpu", verbose=1, tensorboard_log=None)
model.learn(total_timesteps=...)
```
- `MultiInputPolicy` si l'observation est un `Dict` (vecteur + raster).
- Plusieurs envs : `SubprocVecEnv` (le validateur Shapely est le goulot, pas le réseau). **CPU** : MLP/CNN petits, MPS
  n'aide généralement pas.
- Évaluer avec `evaluate_policy(..., use_masks=True)` sur la suite **figée**, pas sur l'env d'entraînement randomisé.

## Points de départ (à ajuster, ce ne sont pas des garanties)
`learning_rate=3e-4`, `n_steps=2048` par env, `batch_size=256`, `n_epochs=10`, `gamma≈0.99–1.0` (épisodes finis),
`gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01` (plus haut si l'entropie s'effondre trop tôt).
Normaliser les récompenses (`VecNormalize(norm_reward=True)`) ; ne pas normaliser le masque.

## Curriculum
1. Petite surface, inventaire réduit, boucle à moitié construite → apprendre à fermer.
2. Agrandir surface/inventaire progressivement tant que le taux de fermeture > seuil (ex. 80 %).
3. Ajouter les ponts seulement après la phase 6, avec un env et un catalogue versionnés.

## Diagnostic
| Symptôme | Causes probables | Action |
|---|---|---|
| Aucune boucle fermée | récompense trop éparse, horizon trop long | curriculum, façonnage par potentiel, démarrage partiel |
| Entropie → 0 très vite | `ent_coef` trop bas, récompense dominée par un comportement trivial | augmenter `ent_coef`, revoir l'échelle |
| `explained_variance` ≈ 0 | observation non informative, retours non normalisés | vérifier l'observation, `VecNormalize` |
| Erreur / blocage sur masque | masque tout faux (impasse) | gérer l'impasse dans l'env (voir `rl-env-design`) |
| Récompense monte, circuits invalides ou courts | reward hacking, tolérance exploitable | auditer avec le validateur complet, resserrer |
| Résultat instable entre seeds | variance normale en RL | ≥ 5 seeds, médiane + IC, ne pas choisir la meilleure seed |

## Hygiène
- Seed global (python, numpy, torch, env) enregistré ; checkpoints réguliers ; arrêt anticipé sur la métrique de la suite figée.
- Chaque run est tracé dans MLflow (voir `rl-benchmark-mlflow`) : hash du catalogue, SHA git, config complète.
- Ne conclure qu'après comparaison aux baselines à budget égal (D12). Un agent qui n'égale pas la LNS n'est pas un échec du
  projet : c'est un résultat.
- Alternatives à tester si (a) stagne : politique-prior pour le beam (b), politique de destruction pour la LNS (c).
