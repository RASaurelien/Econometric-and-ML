# Nowcasting et prévision macroéconomique avec FRED-MD

Ce projet compare des méthodes économétriques classiques et des méthodes d'apprentissage pour prévoir une variable macroéconomique mensuelle (par défaut la production industrielle, `INDPRO`) à partir d'un grand panel d'indicateurs (FRED-MD, environ 120 séries). L'évaluation est faite en pseudo temps réel : à chaque date, les modèles ne voient que les données disponibles jusque-là.

Le sujet est inspiré du livre de Greene (Econometric Analysis) pour la partie séries temporelles et prévision, et des références listées en bas du document.

## Données

On utilise le fichier `current.csv` de FRED-MD (McCracken et Ng), à télécharger sur la page de la Fed de Saint-Louis puis à passer avec `--csv`. Le code lit le format du fichier (la deuxième ligne contient les codes de transformation), applique ces transformations (différences, log-différences, etc.), borne les valeurs extrêmes à 10 écarts interquartiles autour de la médiane et remplace les valeurs manquantes par la dernière valeur connue.

Sans fichier, l'option `--synthetic` génère un panel de test à facteurs avec une cible partiellement non linéaire, utile pour vérifier que le code tourne. Le chargeur a été vérifié sur un faux fichier au même format que `current.csv`, mais pas sur le vrai fichier.

## Modèles

Dans `models.py`, tous avec la même interface `fit` / `predict` :

- AR dont l'ordre est choisi par BIC (benchmark)
- indice de diffusion de Stock et Watson : facteurs principaux du panel puis régression linéaire. C'est la version à deux étapes d'un modèle à facteurs, pas un filtre de Kalman complet
- elastic net sur l'ensemble du panel, avec validation croisée adaptée aux séries temporelles
- LSTM et transformer (PyTorch) sur une fenêtre de 12 mois de facteurs principaux, moyennés sur plusieurs graines

Pour limiter le temps de calcul, les modèles linéaires sont réestimés tous les mois ou tous les 6 mois, les réseaux de neurones tous les 12 mois.

## Évaluation

`evaluate.py` contient la RMSE et le test de Diebold-Mariano (perte quadratique) avec la correction de Harvey, Leybourne et Newbold, qui compare chaque modèle à l'AR. Le script produit un tableau (`results/nowcast_comparaison.csv`) et un graphique du gain cumulé d'erreur quadratique par rapport à l'AR, qui montre à quelles périodes un modèle gagne (souvent pendant les récessions).

## Utilisation

```
pip install numpy pandas scipy scikit-learn matplotlib torch
python main.py --synthetic
python main.py --csv data/current.csv --target INDPRO --start 1990-01-01
python main.py --csv data/current.csv --target CPIAUCSL --h 3 --no_torch
```

## Limites

- Le fichier `current.csv` est un vintage récent : la révision des données n'est pas prise en compte, donc ce n'est pas du vrai temps réel. La page de FRED-MD propose des vintages historiques qui permettraient de le faire.
- Le seuil de valeurs extrêmes (médiane et écart interquartile) est calculé sur tout l'échantillon, ce qui introduit une petite fuite d'information, comme dans l'approche usuelle de la littérature.
- Le test de Diebold-Mariano est fait par modèle contre l'AR. Un Model Confidence Set serait la suite logique pour comparer tous les modèles à la fois.
- Avec quelques centaines d'observations mensuelles, les réseaux de neurones ont peu de données. Ils servent surtout de comparaison, on ne doit pas s'attendre à ce qu'ils battent systématiquement les modèles à facteurs.
- Le code PyTorch (LSTM et transformer) n'a pas été exécuté au moment de l'écriture.

## Références

- McCracken, M. et Ng, S. (2016), « FRED-MD: A Monthly Database for Macroeconomic Research », *Journal of Business & Economic Statistics*. Version working paper : https://doi.org/10.20955/wp.2015.012
- Page des données FRED-MD et FRED-QD (fichier `current.csv`, vintages historiques) : https://www.stlouisfed.org/research/economists/mccracken/fred-databases
- Stock, J. et Watson, M. (2002), « Macroeconomic Forecasting Using Diffusion Indexes », *Journal of Business & Economic Statistics* (indice de diffusion)
- Diebold, F. et Mariano, R. (1995), « Comparing Predictive Accuracy », *Journal of Business & Economic Statistics*
- Harvey, D., Leybourne, S. et Newbold, P. (1997), « Testing the equality of prediction mean squared errors », *International Journal of Forecasting* (correction de petit échantillon)
- Hochreiter, S. et Schmidhuber, J. (1997), « Long Short-Term Memory », *Neural Computation* (LSTM)
- Vaswani et al. (2017), « Attention Is All You Need », NeurIPS (transformer)
