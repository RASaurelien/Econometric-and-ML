# Prévision de la volatilité réalisée : HAR-RV contre apprentissage statistique

Ce projet part des données haute fréquence pour calculer des mesures de volatilité réalisée, puis compare le modèle HAR-RV, qui est le benchmark de la littérature, à des variantes économétriques et à des modèles d'apprentissage (ridge, forêt aléatoire, réseau de neurones, LSTM). L'évaluation est faite en prévision hors échantillon avec une fenêtre glissante.

Le sujet est inspiré du livre de Greene (Econometric Analysis) et des références listées en bas du document.

## Contenu

| fichier | rôle |
|---|---|
| `realized.py` | variance réalisée, semi-variances, variation bipuissance, composante de saut, variables HAR |
| `simulate_intraday.py` | génère des rendements intraday simulés (sert à tester sans données de marché) |
| `models.py` | HAR, SHAR, ridge sur 22 retards, forêt aléatoire, MLP et LSTM (PyTorch) |
| `evaluate.py` | pertes MSE et QLIKE, test de Diebold-Mariano, régression de Mincer-Zarnowitz |
| `main.py` | prévisions glissantes, tableau de comparaison, graphiques |
| `test_realized.py` | vérifications des mesures réalisées et du chargeur de fichiers |

## Modèles

On prévoit le logarithme de la variance réalisée moyenne sur les `h` jours suivants (par défaut `h=1`).

- HAR : régression sur la variance réalisée du jour, de la semaine (5 jours) et du mois (22 jours), en logarithmes (Corsi, 2009).
- SHAR : le terme quotidien est remplacé par les semi-variances négative et positive (Patton et Sheppard, 2015), toujours en logarithmes.
- Ridge sur les 22 derniers log RV quotidiens, sans imposer la structure HAR.
- Forêt aléatoire sur les variables HAR et les semi-variances.
- MLP et LSTM (PyTorch), moyennés sur trois graines, avec early stopping sur la fin de la fenêtre d'apprentissage. Le LSTM lit les 22 derniers jours de (log RV, log semi-variance négative, log semi-variance positive).

Les prévisions en niveau sont obtenues avec la correction log-normale `exp(f + sigma2 / 2)`, où `sigma2` est la variance des résidus du HAR sur la fenêtre d'apprentissage. La même correction est utilisée pour tous les modèles.

## Évaluation

- MSE sur le log de la RV, MSE sur la RV en niveau et QLIKE. Le QLIKE est une perte robuste au bruit de la mesure de volatilité (Patton, 2011), c'est celle qu'il faut regarder en priorité.
- Test de Diebold-Mariano de chaque modèle contre le HAR, sur le QLIKE et sur le MSE en logarithme.
- Régression de Mincer-Zarnowitz (pente et R2).
- Graphique de la volatilité annualisée prévue et du gain cumulé de QLIKE par rapport au HAR.

Les modèles linéaires sont réestimés chaque jour, le ridge tous les 5 jours, la forêt tous les 20 jours et les réseaux de neurones tous les 40 jours.

## Données

Sans option, le script utilise des rendements simulés (`simulate_intraday.py`) : log-variance à deux composantes persistantes, saisonnalité intraday en U, effet de levier et sauts. Comme le levier et les sauts sont écrits dans le simulateur, un avantage du SHAR sur le HAR dans ces simulations est attendu et ne dit rien sur les marchés réels.

Avec des vraies données, `--csv` lit un fichier dont la première colonne contient la date et l'heure et une colonne `price` contient le prix intraday. Le script le rééchantillonne à 5 minutes (`--freq`) et calcule les rendements à l'intérieur de chaque journée. Les estimateurs de `realized.py` sont les plus simples, ils peuvent être remplacés par d'autres (noyaux réalisés, sous-échantillonnage).

## Utilisation

```
pip install numpy pandas scipy scikit-learn statsmodels matplotlib torch
python test_realized.py
python main.py
python main.py --h 5 --window 750
python main.py --csv data/spy_5min.csv --freq 5min
```

## Limites

- Les résultats viennent de simulations par défaut. Aucune conclusion sur de vrais actifs sans passer des vrais prix.
- Le script de chargement de fichiers a été vérifié sur un fichier construit à partir des rendements simulés, pas sur des données de marché. Il ne gère ni les jours fériés partiels ni les trous de prix.
- La littérature discute encore l'effet asymétrique des semi-variances : un article de conférence récent soutient qu'une partie de cet effet vient d'un biais d'estimation lié à la dérive des prix (https://research.bangor.ac.uk/cy/publications/good-and-bad-volatility-estimation-for-drift-diffusion-process/).
- Le MLP et le LSTM n'ont pas été exécutés au moment de l'écriture, seul leur code compile. Leurs hyperparamètres sont fixés à la main.
- Le test de Diebold-Mariano est fait modèle par modèle contre le HAR, sans correction pour les comparaisons multiples.
- Avec `h > 1` les cibles se chevauchent, le test de Diebold-Mariano utilise une variance à longue portée simple (autocovariances jusqu'à `h - 1`).

## Références

- Corsi, F. (2009), « A Simple Approximate Long-Memory Model of Realized Volatility », *Journal of Financial Econometrics* 7(2), 174-196. https://doi.org/10.1093/jjfinec/nbp001
- Patton, A. et Sheppard, K. (2015), « Good Volatility, Bad Volatility: Signed Jumps and the Persistence of Volatility », *Review of Economics and Statistics* 97(3), 683-697. https://doi.org/10.1162/REST_a_00503
- Patton, A. (2011), « Volatility Forecast Comparison using Imperfect Volatility Proxies », *Journal of Econometrics* 160(1), 246-256. Version working paper : https://econpapers.repec.org/RePEc:uts:rpaper:175
- Andersen, T., Bollerslev, T., Diebold, F. et Labys, P. (2003), « Modeling and Forecasting Realized Volatility », *Econometrica* (variance réalisée)
- Barndorff-Nielsen, O. et Shephard, N. (2004), « Power and Bipower Variation with Stochastic Volatility and Jumps », *Journal of Financial Econometrics* (variation bipuissance et sauts)
- Diebold, F. et Mariano, R. (1995), « Comparing Predictive Accuracy », *Journal of Business & Economic Statistics*
- Harvey, D., Leybourne, S. et Newbold, P. (1997), « Testing the equality of prediction mean squared errors », *International Journal of Forecasting*
- Hochreiter, S. et Schmidhuber, J. (1997), « Long Short-Term Memory », *Neural Computation* (LSTM)
