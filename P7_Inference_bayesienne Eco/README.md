# mini-bayes

Une petite librairie d'inférence bayésienne en Python, avec les boucles d'échantillonnage écrites deux fois, en Rust et en C++, et une version Python de référence. Elle propose un échantillonneur de Gibbs pour un panel linéaire à effets aléatoires et un Metropolis-Hastings pour un logit, avec les diagnostics de convergence usuels (R-hat, taille d'échantillon effective). Le point de départ est la partie bayésienne de *Econometric Analysis* de Greene (chaînes de Markov, échantillonnage de Gibbs, Metropolis-Hastings).

Les trois implémentations utilisent le même générateur aléatoire et le même ordre de calcul. À graine égale, elles produisent donc la même chaîne, tirage par tirage.

## Ce qu'il y a dedans

- **Gibbs pour un panel à effets aléatoires** : y_it = x_it'b + a_i + e_it, avec priors normal sur b et inverse-gamma sur les deux variances. Toutes les lois conditionnelles sont exactes.
- **Metropolis-Hastings pour un logit** : marche aléatoire, proposition donnée par l'approximation de Laplace (règle 2.38 / racine de k)
- **Trois cœurs interchangeables** : Rust, C++ et Python. Le choix est automatique (`backend="auto"`) ou forcé.
- **Générateur maison** : xoshiro256** initialisé par SplitMix64, normales par Box-Muller, gamma par Marsaglia-Tsang, identique dans les trois langages
- **Diagnostics** : R-hat découpé, taille d'échantillon effective (règle de Geyer), résumé a posteriori

## Installation

```
pip install -r requirements.txt
make
```

`make` compile les deux cœurs. Il faut `g++` et `cargo` (testé sous Linux, Rust 1.75, aucune dépendance externe). Sans compilation, tout fonctionne avec la version Python, bien plus lente. Seuls numpy et scipy servent à faire tourner la librairie. statsmodels et pytest ne sont là que pour les tests et l'exemple.

## Exemple rapide

```python
import bayes as by

res = by.gibbs_panel(y, X, ids, n_iter=4000, burn=1000)
print(res.resume())

logit = by.mh_logit(y, X, prior_sd=10.0, backend="cpp")
print(logit.resume(), logit.acceptation)
```

`python exemple.py` fait tourner les deux modèles sur des données simulées. Ce qu'on y voit :

- **Panel** : 200 individus observés 6 fois, 12 000 tirages en 0.4 s. Les moyennes a posteriori (0.945, 1.995, -1.019 pour les coefficients, 0.992 pour sigma2, 0.629 pour tau2) sont presque identiques aux estimations REML de MixedLM (0.944, 1.995, -1.019, 0.990, 0.617). Le R-hat est de 1.00 partout.
- **Logit** : les moyennes a posteriori (0.480, 1.047, -0.438, 0.251) sont à moins de 0.01 du maximum de vraisemblance, avec un taux d'acceptation d'environ 0.30 et plus de 5000 tirages effectifs par paramètre.
- **Trois langages, une chaîne** : l'écart maximal entre les chaînes de Gibbs de Rust, de C++ et de Python est de 3e-15.

## Organisation

```
bayes/
    rng.py           générateur, version Python
    reference.py     Gibbs et Metropolis-Hastings en Python (spécification et tests)
    coeurs.py        chargement des cœurs compilés et interface commune
    modeles.py       gibbs_panel et mh_logit
    diagnostics.py   R-hat, ESS, résumé
cpp/bayes.cpp        cœur C++
rust/src/lib.rs      cœur Rust
tests/test_validation.py
Makefile
exemple.py
benchmark.py
```

## Complexité

Notations : n observations, k régresseurs, N individus, I itérations, m chaînes.

| Méthode | Temps | Mémoire |
|---|---|---|
| Gibbs, préparation | O(n k²) | O(n k) |
| Gibbs, une itération | O(n k + k³ + N) | O(N + k²) |
| Gibbs, total | O(n k² + I(n k + k³ + N)) | O(n k + I(k + 2)) |
| Metropolis-Hastings, une itération | O(n k + k²) | O(k) |
| Metropolis-Hastings, total | O(I(n k + k²)) | O(n k + I k) |
| Mode et Laplace du logit | O(iter(n k² + k³)) | O(n k) |
| R-hat et ESS | O(m I log I) | O(m I) |

Le tirage d'un gamma demande en moyenne un peu plus d'une normale. La matrice X'X est calculée une seule fois, ce qui évite de repasser sur les données pour la partie k × k du coefficient.

`python benchmark.py` compare les trois cœurs sur la même chaîne :

- **Gibbs** (n = 5000, 500 individus, 2000 itérations) : 0.16 s en Rust, 0.14 s en C++, 1.8 s pour la référence Python, soit un gain de ×11 à ×13. Les deux cœurs compilés sont à moins de 15 % l'un de l'autre.
- **Metropolis-Hastings** (k = 5, 20 000 itérations) : le gain dépend de n. Il est de ×5 pour n = 200, ×1.1 pour n = 1000, et la version compilée est **plus lente** que numpy pour n = 5000 (2.8 s en Rust, 3.1 s en C++, 1.2 s en Python). Chaque itération évalue exp et log1p pour chaque observation, et numpy les calcule par instructions vectorielles, alors que les cœurs appellent la libm un nombre à la fois. Les cœurs compilés gagnent quand chaque itération fait peu de calcul vectorisable, pas quand elle est dominée par des fonctions transcendantes sur de longs tableaux.

## Validation

```
pytest
```

Les 23 tests couvrent trois niveaux.

**Les trois implémentations coïncident** :

- le générateur : uniformes, normales et gamma (deux formes, dont une inférieure à 1) sont strictement identiques pour quatre graines
- les chaînes de Gibbs de Rust et de C++ sont égales à celle de Python à 1e-10 près (l'écart observé est de 3e-15)
- les chaînes de Metropolis-Hastings sont exactement égales, avec le même nombre d'acceptations
- une même graine redonne la même chaîne, une graine différente non

**Les échantillonneurs sont corrects** :

- chaque loi conditionnelle de Gibbs est comparée à sa formule exacte sur 6000 balayages (moyenne et variance des quantités standardisées)
- le test de Geweke : en alternant simulation des données et balayage de Gibbs, 5 moments des paramètres retombent sur leurs valeurs exactes sous le prior
- pour le logit à un paramètre, la moyenne, l'écart-type et trois quantiles de la chaîne sont comparés au posterior calculé par intégration numérique sur une grille
- pour le logit à 4 paramètres, la moyenne a posteriori est à moins de 0.2 écart-type du maximum de vraisemblance, et les écarts-types sont à 6 % près de ceux de `statsmodels`
- le panel est comparé à MixedLM (coefficients à 0.02, variances à 5 % et 12 %)

**Les diagnostics** : l'ESS d'un AR(1) à rho = 0.9 est à 15 % de sa valeur théorique n(1 - rho)/(1 + rho), et le R-hat détecte des chaînes décalées.

## Limites

- Le panel n'a qu'une constante aléatoire par individu : pas de pentes aléatoires, erreurs gaussiennes. Les priors inverse-gamma par défaut sont peu informatifs sur sigma2, mais celui sur tau2 (IG(1, 1)) pèse quand il y a peu d'individus. Les hyperparamètres se règlent dans `gibbs_panel`.
- Le Metropolis-Hastings utilise une proposition fixe construite à partir de l'approximation de Laplace, sans adaptation. Cela suffit pour un logit à quelques paramètres, mais le mélange se dégrade vite quand k grandit.
- L'identité exacte entre langages dépend de la libm : les trois cœurs s'appuient sur celle du système. Sur une autre plateforme, les écarts de dernier bit peuvent apparaître. Pour Gibbs ils restent de l'ordre de 1e-15. Pour Metropolis-Hastings, une chaîne ne diverge que si une décision d'acceptation tombe exactement à la limite, ce qui n'arrive pas dans les tests.
- Le générateur convient aux simulations, pas à la cryptographie.
- La version Python de référence est lente (de l'ordre de la seconde pour 2000 itérations). Elle sert de spécification et de test, pas de moteur.
- Les R-hat et ESS n'utilisent pas la normalisation par les rangs de Vehtari et al. (2021).

## Références

Greene, *Econometric Analysis* (8e édition) : la partie sur l'inférence bayésienne et les méthodes MCMC. Gelman et al. (2013) pour les modèles hiérarchiques et les diagnostics, Geweke (2004) pour le test de la distribution jointe, Roberts, Gelman et Gilks (1997) pour le taux d'acceptation optimal, Marsaglia et Tsang (2000) pour le tirage gamma, Blackman et Vigna (2018) pour xoshiro, Vehtari et al. (2021) pour R-hat et l'ESS.
