---
name: rl-env-design
description: Use when designing or debugging Slotor's Gymnasium environment for RL track construction — observation and action spaces, action masks from the validator, dead-end handling, potential-based reward shaping, determinism and domain randomization.
---

# Environnement RL de construction de circuit

## Principes
- L'env **réutilise le validateur incrémental** (D6) ; il ne contient aucune règle physique propre.
- Interface Gymnasium : `reset(seed=...)`, `step(a)` → `(obs, reward, terminated, truncated, info)`. Déterministe à seed égal.
- Trois formulations (D11) : (a) construction complète ; (b) politique-prior pour le beam ; (c) politique de destruction pour
  la LNS. Concevoir l'env pour que (a) et (c) partagent le cœur (état, masque, validateur).

## Actions
- Discrètes : un indice par `(type de pièce, miroir)` du catalogue actif + éventuellement `CLOSE` (terminer la boucle).
  Avec ponts : actions montée/descente incluses. Le catalogue fixe la taille de l'espace → garder l'ordre stable.
- **Masque** `action_masks() -> np.ndarray[bool]` : inventaire restant, surface, collision, borne de fermeture admissible,
  `CLOSE` seulement si la pose de fermeture est atteinte à tolérance.
- **Impasse** : si aucune action n'est valide, ne pas renvoyer un masque tout faux (MaskablePPO plante ou boucle).
  Détecter après chaque `step`, terminer l'épisode (`terminated=True`) avec une pénalité modérée, ou prévoir une action
  `ABANDON` toujours valide.

## Observation (taille fixe)
- Vecteur : inventaire restant normalisé par type ; pose relative au connecteur de départ `(dx, dy, sin θ, cos θ, level)`
  dans le repère de la pose courante ; longueur cumulée / borne ; nombre de pièces / max ; temps restant.
- Carte d'occupation : raster `H×W` de la surface (par ex. 64×40), canaux : occupé niveau 0, occupé niveau 1, position courante
  (CNN petit). Normaliser, ne pas fournir de coordonnées absolues non bornées.
- Dict d'observations (`MultiInputPolicy`) si vecteur + image ; sinon aplatir et utiliser un MLP.
- Domain randomization : tirer surface et inventaire à chaque `reset` (distribution documentée), sinon l'agent
  mémorise une instance. Garder une suite d'évaluation **figée** distincte.

## Récompense
- Terminale : `+ longueur(circuit)/référence` si boucle valide fermée ; `0` ou petite pénalité sinon.
- Densité : gain de longueur par pièce posée (même signe que l'objectif).
- Façonnage de fermeture **par potentiel** (Ng et al.) : `r' = r + γΦ(s') − Φ(s)`, avec `Φ(s) = −distance de pose
  à la cible de fermeture`. Un façonnage non basé sur un potentiel peut changer la politique optimale : à éviter.
- Jamais de récompense pour un état invalide ; aucun gain exploitable par tolérance cachée (reward hacking).
- Documenter les échelles ; viser des retours de l'ordre de 1.

## Vérifications obligatoires
1. `gymnasium.utils.env_checker.check_env(env)` passe.
2. **Invariant** : sur ≥ 10 000 épisodes aléatoires *masqués*, tout circuit marqué fermé est accepté par le validateur complet.
3. Le masque n'est jamais tout faux en fin de `step` non terminal.
4. Déterminisme : même seed ⇒ mêmes trajectoires.
5. Profil : temps par `step` ; le validateur domine, viser plusieurs milliers de pas/s par process avant d'entraîner.
6. Épisodes aléatoires masqués = **baseline « aléatoire valide »** du benchmark.

## Pièges
- Horizon long + récompense éparse : commencer par un curriculum (petite surface, peu de pièces, boucle presque faite).
- Observation dépendante de l'ordre des pièces du catalogue : figer l'ordre et versionner (hash du catalogue).
- Indices d'actions qui changent quand l'inventaire change : l'espace d'actions reste constant, seul le masque varie.
