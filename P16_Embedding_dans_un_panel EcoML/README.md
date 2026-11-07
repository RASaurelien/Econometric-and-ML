# Embeddings contre effets fixes en données de panel

Ce projet compare l'estimation d'un coefficient `beta` et la prévision hors échantillon dans un panel quand l'hétérogénéité non observée est traitée de plusieurs façons : effets fixes classiques, effets fixes interactifs, effets aléatoires pénalisés et embeddings appris par un réseau de neurones. Le point de départ est l'idée de remplacer les variables indicatrices d'unité et de période par des vecteurs appris (embeddings), comme on le fait pour les variables catégorielles à forte cardinalité en apprentissage statistique.

Le sujet est inspiré du livre de Greene (Econometric Analysis, chapitres sur les données de panel) et des références listées en bas du document.

## Contenu

| fichier | rôle |
|---|---|
| `dgp.py` | simulation de panels et découpage des cellules en entraînement et test |
| `fe_models.py` | MCO empilés, effets fixes à deux dimensions, effets fixes interactifs de Bai, effets aléatoires (ridge), variables de Mundlak, écarts-types clusterisés avec `statsmodels` |
| `embed_model.py` | modèle à embeddings en PyTorch (additif ou interactif) |
| `main.py` | comparaison sur trois scénarios, tableau et graphique |
| `test_panel.py` | vérifications du code sur des cas à solution connue |

## Scénarios

Le vrai `beta` vaut 1. Le régresseur `x` est corrélé aux effets individuels, donc les MCO empilés sont biaisés.

- `additif` : `y = beta x + alpha_i + gamma_t + e`, 300 unités, 10 périodes.
- `interactif` : on ajoute une composante `lambda_i' f_t` de rang 2, corrélée à `x`, que des effets additifs ne peuvent pas absorber.
- `peu d'obs. par unite` : modèle additif, 1000 unités, 20 périodes, chaque cellule observée avec probabilité 0.15, soit environ trois observations par unité.

20 % des cellules observées sont mises de côté pour mesurer l'erreur de prévision (RMSE test). Le bruit irréductible donne une RMSE de 1.

## Modèles

- MCO empilés.
- Effets fixes individuels et temporels (TWFE), valables aussi pour un panel déséquilibré (projections alternées, vérifiées contre la régression avec variables indicatrices).
- Effets fixes interactifs de Bai (2009), par moindres carrés alternés avec rang 2, qui est le vrai rang du scénario interactif.
- Effets aléatoires : effets individuels et temporels pénalisés en L2, avec la pénalité choisie sur un échantillon de validation. C'est la version linéaire exacte d'un embedding additif avec pénalité.
- Effets aléatoires avec variables de Mundlak (1978) : on ajoute les moyennes de `x` par unité et par période pour corriger la corrélation avec les effets.
- Embeddings en PyTorch : même modèle que les effets aléatoires (avec ou sans Mundlak), plus une version interactive où un vecteur appris par unité et par période est combiné par produit scalaire. Entraînement en batch complet avec Adam, pénalité et nombre d'époques choisis sur un échantillon de validation.

## Résultats observés pendant le développement

Moyenne sur 15 répétitions par scénario (`python main.py --reps 15 --no_torch`). Colonnes : biais de `beta`, RMSE de `beta`, RMSE de prévision sur les cellules test.

| modèle | additif | interactif | peu d'observations |
|---|---|---|---|
| MCO empilés | 0.525, 0.527, 1.317 | 0.857, 0.861, 1.524 | 0.537, 0.538, 1.325 |
| TWFE | -0.001, 0.023, 1.068 | 0.652, 0.668, 1.534 | -0.010, 0.029, 1.273 |
| Effets fixes interactifs (Bai) | 0.002, 0.028, 2.778 | -0.002, 0.020, 1.458 | 0.050, 0.060, 2.440 |
| Effets aléatoires (ridge) | 0.063, 0.080, 1.064 | 0.755, 0.763, 1.481 | 0.234, 0.245, 1.197 |
| Effets aléatoires + Mundlak | 0.000, 0.023, 1.045 | 0.652, 0.668, 1.465 | -0.001, 0.027, 1.180 |

Ce qu'on peut en retenir :

- Les effets additifs échouent dans le scénario interactif : le biais de TWFE est de 0.65, l'estimateur de Bai le corrige. Ce scénario est celui qui justifie un modèle plus flexible.
- Un effet aléatoire pénalisé est biaisé quand `x` est corrélé aux effets individuels (jusqu'à +0.23 avec peu d'observations par unité). La pénalité choisie pour bien prévoir réduit les effets estimés, et `beta` absorbe la différence. Un embedding additif avec pénalité fait le même compromis. Avec les variables de Mundlak, le biais disparaît.
- Quand il y a peu d'observations par unité, les effets aléatoires avec Mundlak prévoient mieux que TWFE (1.180 contre 1.273) pour un `beta` aussi précis. C'est le cas où la régularisation aide.
- L'estimateur de Bai surajuste en prévision dans les scénarios où le rang imposé est trop élevé ou les données trop rares (RMSE test de 2.78 et 2.44), alors que son estimation de `beta` reste bonne. Sur ce simple exemple, le gain de prévision de l'interactif reste faible (1.458 contre 1.465).

Ces chiffres viennent de mes exécutions avec des graines fixées, ils sont à refaire chez soi avant de les citer. Ils ne comprennent pas les modèles PyTorch.

## Les embeddings PyTorch n'ont pas été exécutés

PyTorch n'était pas installé dans l'environnement où j'ai écrit le code, donc `embed_model.py` compile mais n'a jamais tourné, et `main.py` n'a été exécuté qu'avec `--no_torch`. Les trois modèles à embeddings (additif, additif avec Mundlak, interactif avec Mundlak) sont des hypothèses à tester :

- l'embedding additif devrait reproduire les effets aléatoires de la table (même modèle, optimisé par Adam au lieu d'une résolution exacte), et avec Mundlak il devrait donner les résultats de la ligne correspondante ;
- l'embedding interactif avec Mundlak est le seul modèle de la liste qui peut à la fois absorber la corrélation additive et la composante interactive, mais il peut aussi surajuster comme l'estimateur de Bai. Il faut regarder son RMSE test et pas seulement `beta`.

Si les résultats PyTorch s'écartent beaucoup de ces attentes, il faut d'abord vérifier le taux d'apprentissage, le nombre d'époques et la grille de pénalité avant de conclure quoi que ce soit.

## Utilisation

```
pip install numpy scipy pandas scikit-learn statsmodels matplotlib torch
python test_panel.py
python main.py --reps 15
python main.py --no_torch
python main.py --single
```

L'option `--single` affiche un seul jeu de données avec les écarts-types clusterisés par unité (MCO empilés et TWFE avec variables indicatrices, via `statsmodels`).

## Limites

- Données simulées uniquement, aucune application à un vrai panel.
- Le biais des effets aléatoires dépend du choix de la pénalité par validation sur la prévision, pas sur l'estimation de `beta`. Choisir la pénalité pour l'inférence causale est un autre problème.
- Le rang de l'estimateur de Bai est fixé au vrai rang (2). En pratique il faut le choisir (critères d'information, validation croisée).
- Les embeddings n'ont pas d'écarts-types. Une inférence sur `beta` demanderait un bootstrap par unité ou une approche de débiaisage.
- Un seul coefficient et un seul régresseur, sans dynamique (pas de variable endogène retardée).

## Références

- Guo, C. et Berkhahn, F. (2016), « Entity Embeddings of Categorical Variables ». https://arxiv.org/abs/1604.06737
- Bai, J. (2009), « Panel Data Models with Interactive Fixed Effects », *Econometrica* 77(4), 1229-1279. https://doi.org/10.3982/ECTA6135
- Mundlak, Y. (1978), « On the Pooling of Time Series and Cross Section Data », *Econometrica* 46(1), 69-85. https://www.econometricsociety.org/publications/econometrica/1978/01/01/pooling-time-series-and-cross-section-data
- Hausman, J. (1978), « Specification Tests in Econometrics », *Econometrica* (comparaison effets fixes et effets aléatoires)
- Chamberlain, G. (1982), « Multivariate Regression Models for Panel Data », *Journal of Econometrics* (approche des effets aléatoires corrélés)
