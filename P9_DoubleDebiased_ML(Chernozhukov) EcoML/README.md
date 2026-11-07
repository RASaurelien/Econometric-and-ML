# Double/Debiased Machine Learning : econométrie et apprentissage statistique

Ce projet part de l'article de Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey et Robins (2018) sur le Double/Debiased ML. L'idée est d'estimer un effet causal quand il y a beaucoup de variables de contrôle et que la forme fonctionnelle est inconnue, en laissant des algorithmes de ML estimer les fonctions de nuisance tout en gardant une inférence valide sur le paramètre d'intérêt.

## Contenu du dossier

| fichier | rôle |
|---|---|
| `dml_lib.py` | estimateurs DML, learners de nuisance et méthodes de comparaison |
| `simulate.py` | processus générateurs de données, avec les vraies fonctions de nuisance |
| `run_simulation.py` | expériences Monte Carlo, tableaux et graphiques |
| `test_dml.py` | vérifications de la mécanique du code sur des designs à solution connue |
| `application_401k.py` | application sur les données 401(k) |

## Ce que fait le code

Deux modèles sont implémentés dans `dml_lib.py` :

- le modèle partiellement linéaire (PLR), `Y = theta D + g(X) + e`, avec partialling-out et cross-fitting. Le cross-fitting peut être répété plusieurs fois, on garde la médiane des estimations avec la correction de variance associée.
- le modèle à traitement binaire (IRM), avec le score AIPW et un cross-fitting stratifié sur le traitement, qui donne l'effet moyen du traitement (ATE). C'est la même spécification que le score ATE de DoubleML (5 folds, estimateur agrégé).

Les fonctions de nuisance peuvent être estimées par lasso (validation croisée), random forest, gradient boosting, ou un petit réseau de neurones écrit avec PyTorch (early stopping sur un échantillon de validation).

Les méthodes de comparaison sont l'OLS naïf, l'OLS avec tous les contrôles, le post-lasso et le post-double-selection de Belloni, Chernozhukov et Hansen, estimées avec `statsmodels` (erreurs standard robustes HC1).

## Simulations

`simulate.py` contient plusieurs designs :

| scénario | modèle | p | remarque |
|---|---|---|---|
| `plr_smooth` | PLR | 50 | non-linéarité douce, régresseurs corrélés (Toeplitz, rho=0.7) |
| `plr_complex` | PLR | 300 | interactions, seuils, valeur absolue, grande dimension |
| `irm_complex` | IRM | 50 | traitement binaire, effet hétérogène en X1 mais ATE connu |
| `plr_linear`, `irm_linear` | PLR, IRM | 50 | designs linéaires, servent de contrôle du code |

Pour chaque scénario, le script affiche le R2 hors échantillon de chaque learner sur Y|X et D|X (c'est ce qui explique en grande partie le classement des estimateurs), puis le biais, l'écart-type, le RMSE, l'erreur standard moyenne et la couverture de l'intervalle à 95 % sur les répétitions Monte Carlo. Les tableaux et les histogrammes sont sauvegardés dans `results/`.

## Tests

`test_dml.py` vérifie que l'ATE du design IRM vaut bien theta, que l'estimateur sans biais avec les vraies nuisances est bien centré, que le DML avec lasso est sans biais et bien calibré quand le lasso est correctement spécifié (designs linéaires), et que l'OLS naïf est biaisé. Ces tests ne couvrent pas les forêts, le gradient boosting ni le réseau de neurones.

## Application

`application_401k.py` applique le DML IRM à l'effet de l'éligibilité au 401(k) sur le patrimoine financier net (données de Poterba, Venti et Wise, récupérées avec le package `doubleml`). C'est l'exemple classique de la littérature, on peut comparer avec l'OLS et les valeurs publiées.

## Utilisation

```
pip install numpy scipy pandas scikit-learn statsmodels matplotlib joblib torch
python test_dml.py
python run_simulation.py --reps 20 --n 500 --n_jobs 4
python run_simulation.py --scenarios plr_complex --learners lasso rf mlp --n_rep 5
python application_401k.py
```

Le learner `mlp` est ignoré automatiquement si PyTorch n'est pas installé. Avec des forêts ou du gradient boosting et p=300, les simulations sont longues, commencer avec `--reps 20`.

## À regarder dans les résultats

- Dans les designs non linéaires, le DML avec lasso reste biaisé, car le lasso ne peut pas représenter les fonctions de nuisance. Le R2 des nuisances affiché au début de chaque scénario le montre.
- Le DML avec gradient boosting a un biais plus faible mais il dépend de la taille d'échantillon : avec n petit (environ 250), le biais en IRM est important.
- La couverture est estimée sur un nombre fini de répétitions, elle est bruitée avec peu de simulations.

## Limites

- Les hyperparamètres des learners sont fixés à la main, il n'y a pas de réglage imbriqué.
- Le cross-fitting avec K=5 et n petit rend les learners instables, surtout le réseau de neurones.
- L'application 401(k) suppose l'ignorabilité conditionnelle, ce qui est discutable dans ces données.
- Le réseau de neurones PyTorch et le script d'application n'ont pas été exécutés au moment de l'écriture.
