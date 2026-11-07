# Haute dimension : lasso, ridge, elastic net et post-double-selection

Ce projet regarde ce qui se passe en économétrie quand le nombre de régresseurs `p` est grand par rapport au nombre d'observations `n`. Deux questions sont traitées sur données simulées : quel estimateur retrouve les bonnes variables et prévoit bien, et peut-on faire de l'inférence valide sur l'effet d'un traitement `D` quand les contrôles sont sélectionnés par lasso.

Le sujet est inspiré du livre de Greene (Econometric Analysis) et des articles listés dans la section Références.

## Méthodes

Dans `hd_lib.py` :

- lasso avec pénalité choisie par validation croisée
- lasso avec la pénalité "plug-in" de Belloni, Chen, Chernozhukov et Hansen (2012), où la variance du bruit est estimée de façon itérative
- elastic net (validation croisée sur `alpha` et `l1_ratio`)
- ridge
- OLS naïf, post-lasso naïf (sélection sur l'équation de `Y` seulement) et post-double-selection (BCH, 2014), estimés avec `statsmodels` avec erreurs robustes HC1

La pénalité plug-in est une adaptation de la formule de l'article à la paramétrisation de scikit-learn, avec des régresseurs de variance proche de 1. Ce n'est pas une copie de l'implémentation du package R `hdm`.

## Design

Le vecteur `beta` est creux (10 coefficients non nuls qui décroissent lentement), les régresseurs suivent une corrélation de type Toeplitz de paramètre `rho` (0, 0.5 et 0.9), et `D` est confondu avec les mêmes variables que `Y`. Les valeurs par défaut sont `n=500` et `p=600`.

## Expériences

1. Sélection et prévision : précision et rappel du support, erreur sur `beta` et RMSE de prévision hors échantillon, pour chaque méthode et chaque niveau de corrélation.
2. Inférence sur l'effet de `D` : biais, RMSE et couverture de l'intervalle à 95 % pour l'OLS naïf, le post-lasso naïf et le post-double-selection (avec pénalité plug-in ou validation croisée).

Les résultats sont écrits dans `results/` (CSV et un graphique de la précision et du rappel selon `rho`).

## Utilisation

```
pip install numpy scipy pandas scikit-learn statsmodels matplotlib
python main.py --reps 50
python main.py --n 300 --p 1000 --s 5
```

Avec p=600 et la validation croisée, le script est assez lent (plusieurs lassos par répétition). Commencer avec `--reps 10`.

## Points à regarder dans les résultats

- La validation croisée sélectionne beaucoup de variables inutiles (précision faible) alors que la pénalité plug-in est très parcimonieuse. Pour la prévision, la validation croisée est en général meilleure.
- Le plug-in peut manquer des coefficients faibles (rappel inférieur à 1, surtout quand `rho` est élevé), mais le post-double-selection reste peu biaisé, parce que les variables manquées dans l'équation de `Y` sont souvent retrouvées dans celle de `D`. C'est l'intérêt de la double sélection.
- Le post-double-selection avec sélection par validation croisée peut rester biaisé quand `rho` est intermédiaire. La version plug-in est celle recommandée par la théorie.
- Quand `rho` est proche de 1, aucune méthode ne retrouve exactement le support : le lasso choisit une variable parmi un groupe de variables corrélées.

## Limites

- Les valeurs de `n`, `p` et la vitesse de décroissance des coefficients ont été choisies après plusieurs essais pour que le post-double-selection soit bien calibré. Avec un `n` plus petit, il peut rester biaisé en échantillon fini, parce que les coefficients manqués ne sont pas assez petits. Ce n'est pas une contradiction de la théorie, qui demande une parcimonie approximative avec un taux suffisant, mais il faut le savoir en interprétant les chiffres.
- Les résultats viennent de simulations avec un nombre limité de répétitions, les couvertures sont bruitées.
- Pas d'application sur données réelles dans ce dossier.

## Références

- Belloni, A., Chernozhukov, V. et Hansen, C. (2014), « Inference on Treatment Effects after Selection among High-Dimensional Controls », *Review of Economic Studies* 81(2). https://arxiv.org/abs/1201.0224
- Belloni, A., Chen, D., Chernozhukov, V. et Hansen, C. (2012), « Sparse Models and Methods for Optimal Instruments with an Application to Eminent Domain », *Econometrica* 80(6), 2369-2429. https://doi.org/10.3982/ECTA9626 (source de la pénalité plug-in)
- Tibshirani, R. (1996), « Regression Shrinkage and Selection via the Lasso », *JRSS B* (lasso)
- Zou, H. et Hastie, T. (2005), « Regularization and Variable Selection via the Elastic Net », *JRSS B* (elastic net)
- Hoerl, A. et Kennard, R. (1970), « Ridge Regression: Biased Estimation for Nonorthogonal Problems », *Technometrics* (ridge)
