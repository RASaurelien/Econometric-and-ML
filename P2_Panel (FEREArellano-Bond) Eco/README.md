# mini-panel

Une petite librairie d'économétrie des panels en numpy, construite autour des chapitres sur les données de panel de *Econometric Analysis* de Greene. Elle couvre les modèles statiques (effets fixes, effets aléatoires, Mundlak) et le modèle dynamique d'Arellano-Bond, avec les tests qui vont avec. Tous les résultats sont comparés à `linearmodels`.

## Ce qu'il y a dedans

- **MCO empilés** et **between**
- **Effets fixes** (within), pour panels équilibrés ou non
- **Effets aléatoires** (Swamy-Arora)
- **Mundlak** : MCO empilés avec les moyennes individuelles des régresseurs
- **Arellano-Bond** : GMM en différences, une ou deux étapes, correction de Windmeijer, option `collapse` pour limiter le nombre d'instruments
- **Tests** : Hausman, Mundlak robuste, F de présence d'effets individuels, Wald, AR(1) et AR(2) d'Arellano-Bond, J de Hansen

## Installation

```
pip install -r requirements.txt
```

Seuls numpy et scipy servent à faire tourner la librairie. pandas, linearmodels et pytest ne sont là que pour les tests.

## Exemple rapide

```python
import panel as pn
from panel.dynamique import arellano_bond

fe = pn.effets_fixes(y, X, ids, cov="cluster")
re = pn.effets_aleatoires(y, X, ids)
print(pn.hausman(fe, re))

res = arellano_bond(y, X, ids, lags=1, collapse=True)
print(res)
```

`y` et `ids` ont une ligne par observation. Pour Arellano-Bond, les lignes d'un même individu doivent être dans l'ordre du temps et le panel doit être équilibré. Les `X` sont supposés strictement exogènes.

`python exemple.py` lance deux simulations. La première montre des effets individuels corrélés au régresseur : les MCO empilés donnent 1.46 et les effets aléatoires 1.32 pour un vrai coefficient de 1, contre 0.96 pour les effets fixes et Mundlak, et les trois tests rejettent l'exogénéité. La seconde illustre le biais de Nickell sur 200 échantillons (T = 6, vrai rho = 0.5) : les effets fixes donnent 0.28 en moyenne, Arellano-Bond 0.49.

## Organisation

```
panel/
    __init__.py    exports publics de la bibliothèque (import panel as pn)
    outils.py        codes des individus, moyennes, within, classe Resultat
    covariances.py   variance classique et clusterisée
    modeles.py       MCO empilés, between, FE, RE, Mundlak
    dynamique.py     Arellano-Bond, tests AR et Hansen
    diagnostics.py   Hausman, F, Wald, Mundlak robuste
tests/
    test_validation.py
exemple.py
```

## Complexité

Notations : n observations au total, N individus, T périodes, K régresseurs, m colonnes de la matrice d'instruments d'Arellano-Bond, q restrictions pour Wald.

| Méthode | Temps | Mémoire |
|---|---|---|
| MCO empilés (classique ou cluster) | O(nK² + K³) | O(nK) |
| Between | O(nK + NK² + K³) | O(nK) |
| Effets fixes | O(nK² + K³) | O(nK) |
| Effets aléatoires | O(nK² + K³), trois régressions | O(nK) |
| Mundlak | O(nK² + K³) | O(nK) |
| Hausman | O(K³) | O(K²) |
| Wald | O(K³ + qK²) | O(K²) |
| Arellano-Bond, deux étapes | O(NTm² + KNm² + m³) | O(NTm) |

Pour Arellano-Bond, m dépend de T : environ T²/2 instruments sans `collapse`, environ T avec. C'est ce qui rend la méthode coûteuse quand T grandit, et c'est pour ça qu'elle est pensée pour N grand et T petit. Le terme KNm² vient de la correction de Windmeijer, qui demande une boucle sur les K coefficients.

## Validation

```
pytest
```

Les 12 tests comparent mes résultats à `linearmodels` :

- effets fixes (classique, clusterisé, panel déséquilibré), MCO empilés clusterisés, between, effets aléatoires et F de présence d'effets individuels
- Arellano-Bond, en empilant les équations et en passant les mêmes instruments à `IVGMM` : coefficients des deux étapes, écarts-types de la première étape et J de Hansen identiques
- Mundlak : les pentes sont bien égales à celles des effets fixes
- cas particuliers : Arellano-Bond avec T = 3 et sans x redonne l'estimateur d'Anderson-Hsiao, et le biais de Nickell apparaît bien pour les effets fixes

Quelques points à connaître :

- pour les écarts-types clusterisés, j'applique le facteur G/(G-1) × (n-1)/(n-k), celui de Stata. `linearmodels` retrouve les mêmes valeurs avec `group_debias=True`
- la correction de Windmeijer n'a pas de référence directe dans `linearmodels`. Je l'ai vérifiée par simulation : sur 300 échantillons (N = 100, T = 6), l'écart-type empirique de rho est 0.126, l'écart-type corrigé moyen 0.131 et l'écart-type non corrigé 0.110
- le test d'effets aléatoires et le Hausman supposent des erreurs homoscédastiques, utilise le test de Mundlak avec variance clusterisée si ce n'est pas le cas

## Limites

Pas de GMM en système (équations en niveaux), pas de variables prédéterminées ou endogènes autres que y retardé, panels équilibrés seulement pour Arellano-Bond, et les p-values statiques utilisent une loi de Student alors que celles d'Arellano-Bond utilisent la loi normale.

## Références

Greene, *Econometric Analysis* (8e édition) : chapitre 11 (modèles de panel) et chapitre 13 (GMM). Arellano et Bond (1991) pour l'estimateur et les tests AR, Windmeijer (2005) pour la correction des écarts-types en deux étapes.
