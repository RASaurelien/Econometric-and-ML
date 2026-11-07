# Effets hétérogènes de traitement : meta-learners et forêt causale

Ce projet compare plusieurs façons d'estimer l'effet conditionnel moyen du traitement, `tau(x) = E[Y(1) - Y(0) | X = x]`, et pas seulement l'effet moyen. L'objectif est de savoir pour quels individus un traitement marche le mieux, ce qui sert directement au ciblage (campagnes, aides, protocoles).

## Estimateurs

Tout est dans `learners.py`, avec une interface `fit(X, T, Y)` et `predict(X)` :

- S-learner : un seul modèle de `Y` sur `(X, T)`, l'effet est la différence des prédictions.
- T-learner : un modèle par groupe (traités et contrôles).
- X-learner (Künzel et al., 2019) : imputation des effets individuels puis régression, pondérée par le score de propension.
- DR-learner : pseudo-outcome doublement robuste avec cross-fitting, puis régression sur `X`.
- Forêt causale simplifiée : arbres construits sur une moitié de l'échantillon avec l'outcome transformé, effets estimés dans les feuilles avec l'autre moitié (honnêteté). Ce n'est pas l'implémentation complète de Wager et Athey, le critère de découpage est plus simple.
- TARNet (PyTorch) : une représentation commune de `X` puis deux têtes, une par bras de traitement.

## Données et évaluation

`data.py` simule un effet de traitement non linéaire (seuils et sinus), une fonction de base non linéaire et, au choix, un traitement randomisé ou un traitement confondu par `X` (option `--observational`). Comme le vrai `tau(x)` est connu, on peut mesurer :

- la PEHE (racine de l'erreur quadratique moyenne sur `tau`)
- la corrélation de rang entre effet prédit et effet vrai
- l'AUUC, aire sous la courbe de Qini calculée sur un échantillon test randomisé, comme on le ferait avec un vrai A/B test (sans connaître `tau`)
- la calibration par quintile d'effet prédit

Les sorties sont dans `results/` : un tableau et un graphique avec les courbes de Qini et la calibration.

## Utilisation

```
pip install numpy pandas scikit-learn matplotlib scipy torch
python main.py --reps 10
python main.py --observational --reps 10
python main.py --no_torch
```

## Limites

- Les données sont simulées, le classement des méthodes dépend du design (ici l'effet est assez lisse avec un seuil).
- Dans le cas observationnel, le S-learner et le T-learner n'ont aucune correction de la confusion, seuls le X-learner et le DR-learner utilisent le score de propension.
- La forêt causale maison n'a pas d'intervalles de confiance.
- La forêt causale suppose un score de propension constant égal à 0.5 (`e=0.5`). Elle n'est donc valable que dans le cas randomisé, ses résultats avec `--observational` ne sont pas fiables.
- Dans le cas observationnel, le DR-learner peut être instable quand le score de propension estimé est proche de 0 ou de 1 (il est tronqué à [0.05, 0.95]).
