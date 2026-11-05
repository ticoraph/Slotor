---
name: track-geometry-testing
description: Use when writing or testing Slotor's geometry engine or validator — poses (x,y,theta,level), piece connectors, loop closure, lane lengths, Shapely collision checks, two-level bridges (flat 2D), golden circuits and property-based tests.
---

# Géométrie et tests du validateur

## Modèle
- `Pose = (x, y, θ, level)`, mm/radians, `level` entier 0..3 (D30), float64. Une pièce a des connecteurs locaux `(x, y, θ)` et un
  `level_in`/`level_out`. Poser une pièce = composer la pose courante avec le connecteur de sortie ; le niveau d'entrée
  doit égaler le niveau courant.
- Courbe de rayon R, angle signé α depuis la pose `(x, y, θ)` :
  `x' = x + R(sin(θ+α) − sin θ)`, `y' = y − R(cos(θ+α) − cos θ)`, `θ' = θ + α`.
  Le signe de α donne gauche/droite (pièce miroir). Vérifier avec un test, ne pas se fier au signe de mémoire.
- Deux voies : polylignes décalées de ±demi-entraxe depuis la ligne médiane. Longueur intérieure ≠ extérieure en
  courbe (L = R·|α| avec R ± décalage). Les cotes viennent du catalogue.

## Fermeture
- Fermée si l'écart `(Δx, Δy, Δθ mod 2π, Δlevel)` est sous `ToleranceConfig`, avec Δθ ramené dans `(−π, π]`.
- Dérive flottante : sommer les angles comme entiers de l'unité minimale du catalogue quand possible.
- Toute boucle simple (sans croisement) a une somme d'angles signés de ±360°. Avec pont, la somme reste un multiple de 360°.

## Collisions (Shapely ≥ 2)
- Pré-filtre `shapely.STRtree` sur les contours, puis test précis.
- **Pénétration, pas contact** (D24) : éroder chaque contour de `collision_depth_mm / 2` (jointures en onglet) puis
  `a.intersection(b).area > collision_area_mm2`. Les rails voisins, et les rails côte à côte (golden_a), se touchent ou
  s'interpénètrent de moins de 1 mm : `overlaps()`/`intersects()` donneraient des faux positifs. Aucun cas particulier
  pour les voisins de la chaîne.
- **Ponts (D4, D30)** : tout reste 2D à plat. `level` est un entier 0..3 : un pont = 3 éléments pour monter (0→1→2→3), ≥ 1 de
  tablier (niveau 3), 3 pour descendre. Un recouvrement (aire > ε) n'est licite que entre deux pièces pleines
  (`level_in = level_out`) écartées d'au moins 3 niveaux (un tablier sur le sol) ; un élément de rampe n'a rien dessous ni
  dessus ; deux tabliers ne se croisent pas ; un seul étage ; `level` redevient 0 à la fermeture. Pas de hauteur, de pente ni
  de dégagement dans le modèle. Sans pièce de niveau > 0, ce chemin ne s'exécute pas.
- Surface : contour entier ⊂ rectangle réduit de la marge (`contains`, avec tolérance).

## Tests à écrire (dans cet ordre)
1. Unitaires : droite, courbe gauche/droite, miroir, composition = pose attendue (valeurs calculées à la main).
2. Fermeture : ovale simple ; 6 courbes de 60° = tour complet ; somme d'angles.
3. **Négatifs** : chevauchement, sortie de surface, inventaire dépassé, boucle ouverte, quasi-contact juste sous/sur le seuil.
4. **Circuits d'or** : reconstruits en JSON dans `tests/fixtures/` à partir des exemples (`medias/`, jamais committés tels
   quels — D17) ; doivent être valides. Les inventaires des captures Track Architect servent d'oracle de longueur.
5. **Hypothesis** : (a) aucun circuit invalide connu n'est accepté ; (b) ajouter puis retirer une pièce redonne l'état ;
   (c) **validateur incrémental ≡ validateur complet** sur séquences aléatoires ; (d) invariance par rotation/translation
   du circuit entier (si la surface est ignorée).
6. Ponts : tablier (niveau 3) au-dessus d'une pièce de niveau 0 accepté ; deux pièces de même niveau rejetées ; élément de rampe
   croisant n'importe quelle pièce rejetée ; rampe de 2 éléments rejetée (`LEVEL_MISMATCH`) ; boucle finissant surélevée rejetée ;
   le croisement sous un élément de rampe (pont décalé d'un cran) rejeté ; circuit d'or `tests/fixtures/bridge_loop.json` valide.

## Pièges
- Contours : polygone du **corps** du rail (plastique complet, pas seulement les rails métalliques). Les bordures
  optionnelles (Randstreifen) sont **exclues** de l'emprise et des collisions (D20).
- Ne pas accepter un circuit parce que deux points « se touchent presque » : calibrer ε sur les circuits réels.
- Ne jamais remplacer la longueur de ligne médiane par une approximation polygonale.
- Pas de nombre magique : tout seuil dans `ToleranceConfig`.
