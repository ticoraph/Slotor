---
name: combinatorial-search-lns
description: Use when implementing or tuning Slotor's non-RL optimizers — closed-loop seed generation, Beam Search with pruning, Large Neighborhood Search (segment substitution), anytime budgets — and when defining the baselines that RL must beat.
---

# Recherche combinatoire : seeds, Beam Search, LNS

Ces méthodes sont la **référence** : le RL (phase 7) n'a de valeur que s'il les bat ou les accélère (D12).

## Seeds fermés (D10)
- Générer des ovales/stades paramétrés (deux demi-tours + droites) qui consomment l'inventaire et tiennent dans la
  surface. Valider avec le validateur unique. Toujours garder une solution courante valide (mode anytime).
- Plusieurs seeds différents (orientation, proportions) pour diversifier.

## Beam Search constructif
- État : séquence de pièces, pose courante `(x, y, θ, level)`, inventaire restant, occupation. Étendre par chaque
  `(pièce, miroir)` valide.
- **Élagage** (du moins cher au plus cher) : inventaire → surface → collision → **borne de fermeture** :
  distance de pose jusqu'au connecteur de départ ≤ longueur de rail restante exploitable ; angle restant atteignable.
  Les bornes doivent être **admissibles** (ne jamais couper une branche réalisable) ; les tester contre une
  énumération exhaustive sur de petits inventaires.
- Déduplication : clé = pose quantifiée + multiensemble de pièces utilisées + empreinte d'occupation grossière.
- Diversité : garder les K meilleurs *par seau* (par exemple par zone de la surface), pas globalement, sinon le
  beam s'effondre sur des variantes quasi identiques.
- Score de tri : longueur cumulée + terme d'optimisme borné, jamais un terme qui récompense un état invalide.

## LNS (substitution de segments) — optimiseur principal
1. **Destroy** : retirer une fenêtre contiguë de k pièces (k variable), ou toutes les pièces dans une zone, ou des pièces
   choisies au hasard pondéré. Les deux connecteurs libres sont `A` et `B` (pose complète, niveau inclus).
2. **Repair** : trouver une séquence reliant exactement `A` à `B` avec les pièces libérées + stock restant, plus longue
   que la séquence retirée. DFS/beam borné avec les mêmes bornes de fermeture (cible = `B`).
3. **Accept** : strictement plus long (puis, à égalité, moins de pièces). Option : acceptation de type recuit pour
   sortir des optima locaux, avec retour à la meilleure solution.
4. Adapter k et les opérateurs selon le taux de succès (adaptive LNS) ; redémarrage depuis un autre seed si stagnation.
5. Chaque candidat passe par le validateur complet ou incrémental testé équivalent.

## Ponts (phase 6)
- Actions supplémentaires « monter » / « descendre » proposées seulement si l'inventaire de pont le permet ; la pose
  porte `level`. Les destroy/repair doivent préserver l'équilibre montée/descente (niveau de `B` = niveau de `A`).

## Budget et reproductibilité
- Budget en temps mur (10 s / 1 min / 5 min), vérifié dans la boucle ; renvoyer à tout moment la meilleure solution.
- `rng` explicite, résultats déterministes à seed égal. Journaliser la courbe longueur(t) pour le benchmark.
- Ne prétendre à l'optimalité que si c'est prouvé ; sinon « meilleure solution trouvée ».

## Baselines à fournir au benchmark
aléatoire valide (rollouts masqués) · glouton (plus longue pièce valide) · beam · LNS.
