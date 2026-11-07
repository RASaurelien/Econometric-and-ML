# mini-series

Une petite librairie d'économétrie des séries temporelles multivariées en Python, construite autour des chapitres sur les séries temporelles et la cointégration de *Econometric Analysis* de Greene. Elle couvre les VAR, l'identification structurelle (Cholesky et long terme), la cointégration de Johansen avec les modèles à correction d'erreur, et un filtre de Kalman écrit en C++. Les résultats sont comparés à `statsmodels`.

## Ce qu'il y a dedans

- **VAR(p)** : estimation, choix de l'ordre (AIC, BIC, HQ, FPE), stabilité, prévisions, causalité de Granger
- **VAR structurel** : identification de Cholesky et de long terme (Blanchard-Quah), fonctions de réponse impulsionnelle, décomposition de variance, intervalles de confiance par bootstrap
- **Johansen** : tests de la trace et de la valeur propre maximale, valeurs critiques simulées, choix du rang
- **VECM** : estimation de alpha, beta et des Gamma, réécriture en VAR de niveaux pour réutiliser les IRF
- **Filtre de Kalman** en C++ (secours en numpy), valeurs manquantes, lisseur de Rauch-Tung-Striebel, estimation par maximum de vraisemblance, modèles de niveau local et ARMA

## Installation

Sous Linux :

```
pip install -r requirements.txt
make
```

Sous Windows avec MSYS2 UCRT64 et `g++` disponible dans le `PATH`, depuis PowerShell :

```powershell
pip install -r requirements.txt
g++ -O2 -shared -o series\libkalman.dll series\kalman.cpp
```

Le Makefile produit automatiquement une DLL sous Windows et une bibliothèque `.so` sous Linux. Sur Windows, le dossier contenant `g++` doit rester dans le `PATH` pour que Python puisse trouver les DLL d'exécution de MSYS2. Sans compilation, tout fonctionne quand même : le filtre bascule sur la version numpy, bien plus lente. Seuls numpy et scipy servent à faire tourner la librairie. statsmodels et pytest ne sont là que pour les tests.

## Exemple rapide

```python
import series as sr

res = sr.var(Y, 2)
print(res.granger([0], [1]))
reponses = sr.irf_var(res, 8, structure="long_terme")

test = sr.johansen(Y, p=2)
print(test, test.rang())

modele = sr.niveau_local(y)
lisse, _ = modele.lisse()
```

`python exemple.py` fait tourner tout cela sur des données simulées. Ce qu'on y voit :

- **Granger** : y1 cause y2 (F = 22.8, p < 0.0001), mais y3 ne cause pas y1 (p = 0.82), ce qui correspond au processus simulé.
- **Blanchard-Quah** : la matrice d'impact vraie est retrouvée à 0.003 près avec 20 000 observations.
- **Johansen** : un rang de cointégration de 1 est retenu à 5 %. Le vecteur beta estimé est (1, -1.04, 0.51) pour un vrai vecteur (1, -1, 0.5), et alpha (-0.30, 0.10, 0.03) pour (-0.3, 0.1, 0).
- **Kalman avec 20 % de valeurs manquantes** : l'erreur quadratique moyenne du niveau lissé est de 0.33, contre 0.90 pour les observations brutes.

## Organisation

```
series/
    var.py           VAR, critères d'information, Granger
    structurel.py    IRF, FEVD, Cholesky, long terme, bootstrap
    johansen.py      test de Johansen, valeurs critiques, VECM
    kalman.py        filtre, lisseur, estimation, niveau local, ARMA
    kalman.cpp       filtre de Kalman en C++
tests/
    test_validation.py
Makefile
exemple.py
benchmark.py
```

## Complexité

Notations : n observations, k variables, p retards, H horizon des IRF, B tirages bootstrap, m états, q variables observées, I itérations de l'optimiseur.

| Méthode | Temps | Mémoire |
|---|---|---|
| VAR(p) par MCO | O(n k²p² + k³p³) | O(n kp) |
| IRF et décomposition de variance | O(H p k³) | O(H k²) |
| IRF avec bootstrap | O(B(T p k² + n k²p² + k³p³ + H p k³)) | O(B T k) |
| Johansen et VECM | O(n k²p² + k³) | O(n kp) |
| Valeurs critiques simulées (une fois par dimension d) | O(reps T d² + reps d³) | O(1000 T d) |
| Filtre de Kalman | O(n(m³ + q m² + q³)) | O(m²), ou O(n m²) si on garde les états |
| Lisseur | O(n m³) | O(n m²) |
| Estimation par MV | O(I p_param n(m³ + q m² + q³)) | O(m²) |

Pour le Kalman, le coût par date vient des produits de matrices m × m. L'estimation par maximum de vraisemblance appelle le filtre à chaque évaluation, d'où l'intérêt du C++ : `python benchmark.py` donne un gain de ×70 pour 1000 observations, ×120 pour 10 000 et ×150 pour 100 000 (4.1 s en numpy contre 0.03 s en C++, pour 4 états et 2 variables observées). Les valeurs critiques de Johansen sont simulées une seule fois par dimension puis gardées en mémoire (environ 1 s).

## Validation

```
pytest
```

Les 19 tests comparent mes résultats à `statsmodels` :

- **VAR** : coefficients, écarts-types, matrice de variance des résidus, prévisions, stabilité, critères d'information, tests de Granger, IRF et décomposition de variance de Cholesky. Les écarts sont de l'ordre de 1e-15.
- **Johansen** : valeurs propres, statistiques de la trace et de la valeur propre maximale (`coint_johansen`), avec ou sans constante. Le VECM redonne alpha, beta, Gamma, la constante et la matrice de covariance de `VECM`.
- **Valeurs critiques** : les valeurs simulées sont à moins de 0.4 des tables tabulées de `statsmodels`.
- **Filtre de Kalman** : log-vraisemblance, états filtrés, prédits et lissés et leurs covariances, avec valeurs manquantes et constantes, comparés à `KalmanSmoother`. Les versions C++ et numpy donnent les mêmes résultats à 1e-10.
- **Modèles estimés** : l'ARMA(1,1) redonne les paramètres et la log-vraisemblance d'`ARIMA`, le niveau local redonne ceux d'`UnobservedComponents`.

Deux vérifications se font par simulation, faute de référence directe :

- **Blanchard-Quah** : avec 40 000 observations, la matrice d'impact vraie est retrouvée à 0.03 près, et l'effet de long terme est bien triangulaire.
- **Bootstrap des IRF** : sur 60 échantillons simulés, les intervalles à 90 % contiennent la vraie réponse dans 80 à 98 % des cas (la couverture n'est pas exactement de 90 %, c'est un percentile simple sans correction de biais).

## Limites

- Johansen : deux cas seulement, sans terme déterministe ou avec une constante libre. Pas de constante ni de tendance dans la relation de cointégration. Avec constante libre, les valeurs critiques sont simulées avec des marches aléatoires à dérive, ce qui reproduit les tables usuelles.
- Pas d'écarts-types pour alpha et les Gamma du VECM.
- SVAR : Cholesky et long terme uniquement. Pas de restrictions de court terme générales (matrices A et B) ni de restrictions de signe.
- Kalman : matrices constantes dans le temps. Pour l'initialisation diffuse j'utilise une grande variance (1e7), pas la version exacte. Les paramètres estimés sont les mêmes que ceux de `statsmodels`, mais la log-vraisemblance diffère d'une constante.
- Dans l'estimation de l'ARMA, la stationnarité est imposée par une pénalité, pas par une reparamétrisation.
- VAR avec constante ou sans terme déterministe, pas de tendance.

## Références

Greene, *Econometric Analysis* (8e édition) : les parties sur les séries temporelles, les données non stationnaires et la cointégration. Lütkepohl (2005) pour les VAR, les IRF et le bootstrap, Johansen (1995) pour la cointégration, Blanchard et Quah (1989) pour l'identification de long terme, Durbin et Koopman (2012) pour le filtre de Kalman.
