# mini-econometrie

Une petite librairie d'économétrie avec principalement numpy, en suivant *Econometric Analysis* de Greene. Le but était de bien comprendre ce qui se passe derrière un `sm.OLS(...).fit()`, donc j'ai tout refait, puis comparé les résultats à statsmodels et linearmodels.

## Ce qu'il y a dedans

- **MCO** avec plusieurs écarts-types : classique, White (HC0 à HC3), Newey-West (HAC) et clusterisés
- **MCG** quand les variances sont connues, et **MCGF** quand on doit les estimer (hétéroscédasticité multiplicative)
- **Doubles moindres carrés (2SLS)** pour les variables instrumentales
- **GMM** en deux étapes, robuste à l'hétéroscédasticité
- **Tests** : Wald, Breusch-Pagan, Hausman, J de Hansen

## Installation

```
pip install -r requirements.txt
```

Seul numpy et scipy servent à faire tourner la librairie. Le reste (statsmodels, linearmodels, pytest) n'est là que pour les tests.

## Exemple rapide

```python
from econometrie import mco, mc2e, hausman

res_mco = mco(y, x)                      # MCO classique
res_rob = mco(y, x, cov="HC3")           # écarts-types robustes
res_iv = mc2e(y, x, z)                   # x instrumenté par z

print(res_iv)
print(hausman(res_iv, res_mco, indices=[1]))
```

`python exemple.py` lance une simulation complète : un régresseur endogène, le biais du MCO, la correction par 2SLS et GMM, puis les tests de Hausman et de Hansen. Sur 300 échantillons, le MCO donne 2.28 en moyenne pour un vrai coefficient de 2, contre 2.00 pour le 2SLS et le GMM.

## Organisation

```
econometrie/
    covariances.py   estimateurs de variance (HC, HAC, cluster)
    modeles.py       MCO, MCG, MCGF, 2SLS, GMM
    diagnostics.py   tests statistiques
tests/
    test_validation.py   comparaison avec statsmodels et linearmodels
exemple.py
```

## Complexité

Notations : n observations, k régresseurs, m colonnes de Z (instruments et exogènes), L retards pour le HAC, G clusters, q restrictions pour Wald.

| Méthode | Temps | Mémoire |
|---|---|---|
| MCO (écarts-types classiques) | O(nk² + k³) | O(nk) |
| MCO avec HC0 à HC3 | O(nk² + k³) | O(nk) |
| MCO avec HAC (Newey-West) | O(Lnk² + k³) | O(nk) |
| MCO avec cluster | O(Gn + nk² + k³) | O(nk) |
| MCG | O(nk² + k³) | O(nk) |
| MCGF | O(nk² + k³), deux passes de MCO | O(nk) |
| 2SLS | O(nm² + nmk + m³ + nk²) | O(nm) |
| GMM deux étapes | O(nm² + nmk + m³ + k³) | O(nm) |
| Wald | O(k³ + qk²) | O(k²) |
| Breusch-Pagan | O(nk²) | O(nk) |
| Hausman | O(k³) | O(k²) |
| J de Hansen | O(nm² + m³) | O(nm) |

Le coût vient surtout de X'X, donc de n × k². Quand n est grand devant k, tout est linéaire en n. Deux points à savoir : la boucle sur les clusters refait un masque de taille n à chaque groupe (d'où le Gn), ce qui peut être amélioré avec un tri préalable, et le HAC coûte L fois plus cher qu'un sandwich simple.

## Validation

```
pytest
```

Les 13 tests comparent mes coefficients, écarts-types et statistiques de test à ceux de statsmodels (MCO, MCG, Wald, Breusch-Pagan) et de linearmodels (2SLS, GMM). Ils passent tous.

Deux détails à connaître :

- pour le J de Hansen, linearmodels garde les poids de la première étape alors que je recalcule la matrice avec les résidus finaux, donc on a un petit écart (environ 0.1 %), c'est pour ça que le test utilise une tolérance
- le MCGF suppose une variance de la forme exp(a + b x), s'il y a un autre type d'hétéroscédasticité il sera moins efficace

## Limites

Pas de données de panel ni de séries temporelles ici (ce sera l'objet d'autres projets), les p-values utilisent toujours une loi de Student à n-k degrés de liberté, et le code n'est pas optimisé pour de très gros jeux de données.

## Références

Greene, *Econometric Analysis* (numéros de la 8e édition) : chapitres 2 à 5 (MCO et inférence), 8 (endogénéité et variables instrumentales), 9 (MCG et hétéroscédasticité) et 13 (GMM).
