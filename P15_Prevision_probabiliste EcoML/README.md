# Prévision conforme pour séries temporelles : couverture empirique sous rupture

Ce projet compare des intervalles de prédiction pour une série temporelle et vérifie leur couverture empirique. L'idée est qu'un intervalle à 90 % doit contenir la vraie valeur 90 % du temps. La prévision conforme donne cette garantie sans hypothèse sur la loi des erreurs, mais seulement si les données sont échangeables, ce qui est faux pour une série qui change de régime. Le projet mesure l'écart et teste une correction en ligne.

Le sujet est inspiré du livre de Greene (Econometric Analysis) pour la partie séries temporelles et des articles listés en bas du document.

## Contenu

| fichier | rôle |
|---|---|
| `data.py` | série simulée : AR non linéaire, volatilité à deux régimes, erreurs de Student, rupture à 70 % de l'échantillon |
| `forecasters.py` | prévision ponctuelle (AR ou gradient boosting) et régressions quantiles pour CQR |
| `conformal.py` | quantile conforme, intervalle gaussien, conforme fixe, conforme glissant, ACI |
| `evaluate.py` | couverture globale et par régime, largeur, score d'intervalle, couverture glissante |
| `main.py` | comparaison sur plusieurs séries simulées ou sur un fichier, tableau et graphiques |
| `test_conformal.py` | tests du quantile, de la couverture et de la garantie d'ACI |

## Méthodes

Le modèle de base est estimé une seule fois sur les 35 % premiers points. Les 15 % suivants servent à la calibration, le reste est le test, parcouru dans l'ordre temporel. Chaque méthode ne voit à la date `t` que les scores des dates précédentes.

- Gaussien : prévision plus ou moins 1.645 écart-type des résidus de calibration (référence naïve).
- Conforme fixe : quantile des résidus absolus de calibration, avec la correction de taille finie. C'est la méthode "split conformal".
- Conforme glissant : même chose, mais le quantile est recalculé sur les 200 derniers scores observés.
- ACI : conforme glissant avec le niveau `alpha_t` ajusté après chaque observation (augmenté si l'intervalle a couvert, diminué sinon).
- CQR fixe : régression quantile conforme, les bornes viennent de deux régressions quantiles (gradient boosting) corrigées par le quantile des scores de calibration.
- CQR + ACI : CQR avec ajustement en ligne du niveau.

Avec un niveau `alpha_t` qui descend trop bas, l'intervalle devient infini. Le code garde les bornes infinies dans les intervalles et remplace l'infini par une grande valeur seulement pour calculer largeurs et scores. La colonne "part infinie" donne la fréquence de ce cas.

## Résultats observés pendant le développement

Moyenne sur 10 séries simulées (T=4000, cible de couverture 0.90, `python main.py --reps 10`) :

| méthode | couverture | couv. calme | couv. agité | largeur | score d'intervalle | pire écart sur 200 obs. |
|---|---|---|---|---|---|---|
| Gaussien | 0.753 | 0.925 | 0.602 | 3.64 | 11.74 | 0.448 |
| Conforme fixe | 0.747 | 0.920 | 0.594 | 3.56 | 11.88 | 0.458 |
| Conforme glissant | 0.879 | 0.975 | 0.797 | 5.92 | 9.21 | 0.153 |
| ACI | 0.900 | 0.970 | 0.840 | 6.42 | 9.09 | 0.088 |
| CQR fixe | 0.788 | 0.918 | 0.674 | 3.74 | 9.66 | 0.312 |
| CQR + ACI | 0.900 | 0.963 | 0.846 | 5.79 | 8.29 | 0.084 |

- Les méthodes fixes sous-couvrent nettement après la rupture, parce que la volatilité double environ. C'est le cas où l'hypothèse d'échangeabilité casse.
- L'ajustement en ligne ramène la couverture moyenne à 0.90. CQR + ACI donne le meilleur score d'intervalle et des intervalles plus étroits que ACI avec le résidu absolu.
- La couverture reste inégale entre régimes : 0.96 en régime calme contre 0.85 en régime agité pour CQR + ACI. ACI garantit la couverture moyenne dans le temps, pas la couverture conditionnelle au régime.

Ces résultats viennent de mes exécutions avec des graines fixées et sont à refaire chez soi avant de les citer. Le simulateur contient la rupture et le changement de régime, donc l'avantage des méthodes adaptatives est en partie construit.

## Tests

`test_conformal.py` vérifie en particulier que le quantile conforme suit la correction de taille finie, que le conforme fixe couvre bien des données échangeables, que ACI avec `gamma = 0` redonne le conforme glissant, que la borne de couverture à long terme d'ACI est respectée même quand la variance est multipliée par 4, et que le conforme fixe échoue sous rupture là où ACI se rétablit.

## Utilisation

```
pip install numpy scipy pandas scikit-learn matplotlib
python test_conformal.py
python main.py --reps 10
python main.py --base ar --no_shift
python main.py --csv data/ma_serie.csv --column rendement
```

Avec `--csv`, le script prend une seule série, la colonne indiquée, et les colonnes de couverture par régime n'ont pas de sens (il n'y a qu'un seul régime étiqueté).

## Limites

- Le modèle de base n'est pas réestimé dans le temps. Dans un vrai usage on le réestimerait régulièrement, ce qui change les résidus et peut réduire le besoin d'adaptation.
- Il y a un seul pas d'avance. Pour un horizon plus long, les erreurs sont corrélées et il faut adapter la méthode.
- Les intervalles sont symétriques autour de la prévision (sauf CQR). Pas de comparaison avec EnbPI ni avec les variantes pondérées ou à fenêtre adaptative.
- La valeur de `gamma` (0.005) et la fenêtre (200) ont été fixées sans réglage. Une valeur plus grande réagit plus vite mais produit des intervalles plus instables.
- Pas de données réelles testées. L'option `--csv` a été vérifiée sur une série simulée enregistrée dans un fichier.

## Références

- Gibbs, I. et Candès, E. (2021), « Adaptive Conformal Inference Under Distribution Shift », NeurIPS 2021. https://arxiv.org/abs/2106.00170
- Gibbs, I. et Candès, E. (2024), « Conformal Inference for Online Prediction with Arbitrary Distribution Shifts », *Journal of Machine Learning Research* (extension d'ACI avec pas adaptatif, non implémentée ici)
- Romano, Y., Patterson, E. et Candès, E. (2019), « Conformalized Quantile Regression », NeurIPS 2019. https://arxiv.org/abs/1905.03222
- Angelopoulos, A. et Bates, S. (2021), « A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification ». https://arxiv.org/abs/2107.07511
- Vovk, V., Gammerman, A. et Shafer, G. (2005), *Algorithmic Learning in a Random World*, Springer (origine de la prévision conforme)
- Gneiting, T. et Raftery, A. (2007), « Strictly Proper Scoring Rules, Prediction, and Estimation », *JASA* (score d'intervalle)
