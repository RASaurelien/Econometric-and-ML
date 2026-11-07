# mini-mle

Une petite librairie de maximum de vraisemblance et de modèles à variable dépendante limitée en numpy, construite autour des chapitres sur le MV et les choix discrets de *Econometric Analysis* de Greene. Tout part d'un seul moteur d'estimation (Newton-Raphson et BFGS maison) sur lequel les modèles s'appuient. Les résultats sont comparés à `statsmodels`.

## Ce qu'il y a dedans

- **Logit et probit** avec scores et hessien analytiques
- **Logit et probit ordonnés**, seuils estimés avec leur écart-type
- **Logit multinomial**
- **Tobit**, censure à gauche, à droite ou les deux
- **Heckman** (sélection d'échantillon) par maximum de vraisemblance
- **Effets marginaux** moyens ou en la moyenne, écarts-types par delta method, pour tous les modèles ci-dessus
- **Variances** : hessien, OPG, robuste (sandwich) et clusterisée
- **Tests** : Wald et rapport de vraisemblance

## Installation

```
pip install -r requirements.txt
```

Seuls numpy et scipy servent à faire tourner la librairie. statsmodels et pytest ne sont là que pour les tests.

## Exemple rapide

```python
import mle

res = mle.logit(y, X, cov="robuste")
print(res)
print(mle.effets_marginaux(res))

tob = mle.tobit(y, X, gauche=0)
hk = mle.heckman(y, X, selectionne, Z)
```

`python exemple.py` fait tourner chaque modèle sur des données simulées. Deux résultats à retenir :

- **Tobit** : avec 35 % d'observations censurées en 0, les MCO donnent 1.30 pour un vrai coefficient latent de 2. Le Tobit retrouve 2.00, et son effet marginal sur la variable observée (1.29) est proche de la pente des MCO, qui estime justement cet effet.
- **Heckman** : sur l'échantillon sélectionné, les MCO donnent 1.33 et 1.89 pour des vrais coefficients de 1 et 2. Le Heckman retrouve 1.03 et 2.01, avec rho estimé à 0.59 pour un vrai rho de 0.6.

## Organisation

```
mle/
    optimiseurs.py   Newton-Raphson, BFGS, recherche de pas
    estimation.py    dérivées numériques, estimer(), résultats, Wald et rapport de vraisemblance
    modeles.py       logit, probit, ordonné, multinomial, Tobit, Heckman
    effets.py        effets marginaux et delta method
tests/
    test_validation.py
exemple.py
```

Pour ajouter un modèle, il suffit d'écrire sa log-vraisemblance par observation et de la passer à `estimer()`. Les scores et le hessien sont alors calculés numériquement, sauf si on les fournit.

## Complexité

Notations : n observations, K régresseurs, J catégories, p paramètres estimés, I itérations de l'optimiseur.

| Méthode | Temps | Mémoire |
|---|---|---|
| Logit, probit (Newton, dérivées analytiques) | O(I(nK² + K³)) | O(nK) |
| Ordonné, Tobit, Heckman (BFGS puis Newton, dérivées numériques) | O(I p n K + p² n K) | O(np) |
| Multinomial, avec p = (J-1)K | O(I p n K J + p² n K J) | O(np) |
| Effets marginaux et delta method | O(p n K²) | O(nK) |
| Wald | O(p³) | O(p²) |

Une évaluation de la log-vraisemblance coûte O(nK) (O(nKJ) pour le multinomial). Un gradient numérique en demande 2p, d'où le terme p n K, et un hessien numérique en demande de l'ordre de 2p², d'où p² n K. Ce hessien est recalculé à chaque itération de Newton finale et une fois pour la variance. Pour le logit et le probit, Newton converge en général en 4 à 6 itérations. Pour le multinomial, p grandit avec J, donc le coût monte vite avec le nombre de catégories.

## Validation

```
pytest
```

Les 14 tests passent. Ce qui est comparé à `statsmodels` :

- logit, probit : coefficients, écarts-types, log-vraisemblance, pseudo R2 et variance robuste, à 1e-6
- effets marginaux du logit, du probit et du multinomial, écarts-types compris
- logit ordonné (`OrderedModel`) et multinomial (`MNLogit`)
- tests du rapport de vraisemblance et de Wald

Pour le reste :

- **Tobit** : comparé à une vraisemblance réécrite dans `GenericLikelihoodModel`. Cette comparaison vérifie l'optimiseur et la variance, mais pas la formule elle-même, puisque c'est la même. La formule est vérifiée autrement : sans censure on retrouve exactement les MCO (sigma est l'écart-type du MV), et l'effet marginal sur E[y] colle à la formule fermée b × Φ(x'b/σ).
- **Heckman** : je n'ai pas trouvé de référence qui donne le MV complet. Il est validé par simulation : sur 20 000 observations, les 7 paramètres sont retrouvés à 0.06 près, alors que les MCO sur l'échantillon sélectionné sont biaisés.
- **Newton et BFGS** arrivent au même optimum depuis un point de départ éloigné, et le hessien numérique redonne les écarts-types du hessien analytique.

## Limites

Pas de modèles de comptage, pas de Heckman en deux étapes. Les dérivées sont numériques pour tous les modèles sauf logit et probit, ce qui donne une précision de l'ordre de 1e-5 sur les écarts-types. Pour le Heckman, la restriction d'exclusion (une variable dans Z absente de X) est à la charge de l'utilisateur. Pour l'ordonné et le multinomial, les effets marginaux se calculent catégorie par catégorie avec `categorie=`. Les p-values utilisent toujours la loi normale.

## Références

Greene, *Econometric Analysis* (numéros de la 8e édition) : chapitre 14 (maximum de vraisemblance), 17 (choix binaires), 18 (choix multinomiaux et ordonnés) et 19 (variables dépendantes limitées : Tobit et sélection d'échantillon). Heckman (1979) pour le modèle de sélection.
