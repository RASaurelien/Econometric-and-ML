# mini-quantile

Une petite librairie de régression quantile et d'estimation non paramétrique en Python, construite autour des parties de *Econometric Analysis* de Greene sur la régression quantile et les méthodes semiparamétriques. Elle regroupe trois familles d'outils : la régression quantile résolue par programmation linéaire, les estimateurs à noyau (densité, Nadaraya-Watson, polynômes locaux) avec choix de la fenêtre par validation croisée, et le modèle partiellement linéaire de Robinson. Les résultats sont comparés à `statsmodels`.

## Ce qu'il y a dedans

- **Régression quantile** : résolution exacte par programmation linéaire, trois estimateurs de variance (sandwich de Powell, erreurs iid, bootstrap par paires), pseudo R2 de Koenker et Machado
- **Plusieurs quantiles** : prévision des quantiles conditionnels avec réordonnancement pour supprimer les croisements, test d'égalité des pentes entre quantiles
- **Densité à noyau** : quatre noyaux, fenêtres de Silverman et de Scott
- **Régression non paramétrique** : Nadaraya-Watson et polynômes locaux de tout degré, intervalles de confiance ponctuels robustes à l'hétéroscédasticité
- **Choix de la fenêtre** par validation croisée leave-one-out, sans refaire les estimations
- **Modèle partiellement linéaire** de Robinson : coefficients à vitesse racine de n et partie non paramétrique libre

## Installation

```
pip install -r requirements.txt
```

Seuls numpy et scipy servent à faire tourner la librairie. statsmodels et pytest ne sont là que pour les tests et l'exemple.

## Exemple rapide

```python
import quantnp as qn

res = qn.regression_quantile(y, x, tau=0.9)
print(res)
print(qn.egalite_pentes(y, x, taus=(0.1, 0.9)))

h = qn.choisir_h(x, y, degre=1)
ajuste, bas, haut = qn.intervalle(x, y, grille, h)

rob = qn.robinson(y, X, z)
```

`python exemple.py` fait tourner tout cela, en commençant par la courbe d'Engel (dépense alimentaire de 235 ménages belges). Ce qu'on y voit :

- **Quantiles d'Engel** : la pente du revenu passe de 0.40 au quantile 0.1 à 0.69 au quantile 0.9, alors que les MCO donnent une seule pente de 0.49. Le test d'égalité des pentes entre ces deux quantiles rejette fortement (chi2 = 31.7).
- **Effet de bord** : sur une vraie droite, en x = 0, Nadaraya-Watson a un biais de 0.24, contre -0.004 pour le linéaire local.
- **Robinson** : quand x dépend de z et que z agit de façon non linéaire sur y, les MCO sans g(z) donnent 1.01 pour un vrai coefficient de 1.5. Robinson donne 1.47.

## Organisation

```
quantnp/
    quantile.py   régression quantile, variances, réordonnancement, test d'égalité des pentes
    noyaux.py     noyaux usuels
    lisseurs.py   densité, polynômes locaux, validation croisée, intervalles
    semipar.py    modèle partiellement linéaire de Robinson
tests/
    test_validation.py
exemple.py
benchmark.py
```

## Complexité

Notations : n observations, k régresseurs, G points de grille, d degré du polynôme local, B tirages bootstrap, q quantiles testés, nh fenêtres essayées, I itérations du solveur.

| Méthode | Temps | Mémoire |
|---|---|---|
| Régression quantile (dual, points intérieurs) | O(I n k²) | O(nk) |
| Variance robuste ou iid | O(n k²) en plus | O(nk) |
| Variance bootstrap | B régressions | O(nk) |
| Test d'égalité des pentes | O(B q) régressions | O(nk) |
| Densité à noyau | O(G n) | O(G n) |
| Polynôme local en G points | O(G n (d+1)²) | O(G n) |
| Validation croisée, une fenêtre | O(n² (d+1)²) | O(n) par bloc |
| Choix de la fenêtre | O(nh n² (d+1)²) | O(n) par bloc |
| Intervalles de confiance | O(n² (d+1)²) | O(G n) |
| Robinson | O(n² (d+1)²) | O(n²) |

La validation croisée reste quadratique en n : `python benchmark.py` donne 0.05 s pour n = 500, 0.8 s pour 2000 et 3.3 s pour 4000, soit un facteur 4 chaque fois que n double. Les calculs se font par blocs de points pour borner la mémoire des tableaux intermédiaires.

Pour la régression quantile, on résout le dual du programme linéaire, qui n'a que k contraintes au lieu de n, avec le solveur HiGHS (points intérieurs au-delà de 20 000 observations). Il faut 0.01 s pour n = 1000, 0.1 s pour 10 000 et 0.8 s pour 100 000. Le `QuantReg` de statsmodels, qui procède par moindres carrés repondérés itérés, est plus rapide à grand n (0.3 s pour 100 000), mais il ne donne qu'une solution approchée (écart de l'ordre de 1e-5 sur les coefficients d'Engel), alors que le programme linéaire donne la solution exacte.

## Validation

```
pytest
```

Les 28 tests couvrent deux niveaux.

**Régression quantile**, sur les données d'Engel :

- coefficients, écarts-types iid et pseudo R2 comparés à `QuantReg` pour cinq quantiles
- écarts-types robustes comparés à `QuantReg` à 3 % près (voir les limites)
- le programme linéaire a une perte au plus égale à celle de `QuantReg` et à celle de 200 perturbations aléatoires, et la médiane est au moins aussi bonne qu'une minimisation directe des écarts absolus
- sur un modèle à position et échelle simulé (20 000 observations), les coefficients aux quantiles 0.1, 0.5 et 0.9 retrouvent les valeurs théoriques à 0.06 près
- le réordonnancement supprime bien les croisements, le bootstrap donne des écarts-types proches du sandwich, et le test d'égalité des pentes rejette avec des erreurs hétéroscédastiques mais pas avec des erreurs homoscédastiques

**Non paramétrique** :

- fenêtres de Silverman et de Scott identiques à `statsmodels`, densité gaussienne identique à `KDEUnivariate`, et les quatre noyaux donnent des densités positives d'intégrale 1
- Nadaraya-Watson et linéaire local identiques à `KernelReg` à 1e-7, et le linéaire local reproduit exactement les droites
- le raccourci leave-one-out e / (1 - L_ii) donne exactement les mêmes erreurs qu'une boucle qui refait chaque estimation
- les fenêtres choisies par validation croisée sont à 1 % de celles de `KernelReg(bw="cv_ls")`
- sur 150 échantillons simulés, les intervalles ponctuels à 95 % couvrent la vraie valeur dans 88 à 99 % des cas (ils ignorent le biais de lissage)
- sur 120 échantillons, Robinson a un biais inférieur à 0.03, une couverture entre 88 et 99 % et un écart-type moyen proche de l'écart-type empirique, alors que les MCO naïfs sont biaisés de plus de 0.1

## Limites

- Le sandwich de Powell dépend du traitement des k résidus exactement nuls (les points d'interpolation du programme linéaire). Je les classe comme négatifs pour que le résultat soit déterministe, `QuantReg` les classe selon le bruit de son algorithme itératif. L'écart est de l'ordre de 1 % sur les écarts-types.
- Un quantile à la fois : les quantiles ne sont pas estimés conjointement, d'où le réordonnancement après coup.
- Les estimateurs non paramétriques n'ont qu'une variable explicative. Pas de bandes de confiance uniformes, et les intervalles ponctuels ignorent le biais de lissage.
- Avec un noyau à support compact, un point sans observation dans son voisinage donne `nan`.
- Robinson : z est scalaire, la fenêtre par défaut est fixe (écart-type de z fois n^(-1/3)) et il n'y a pas de troncature des observations à faible densité.
- Le bootstrap par paires ne tient pas compte d'une éventuelle dépendance entre observations.

## Références

Koenker et Bassett (1978) pour la régression quantile, Koenker (2005) pour le livre de référence, Powell (1986) et Hall et Sheather (1988) pour la variance, Chernozhukov, Fernandez-Val et Galichon (2010) pour le réordonnancement, Nadaraya (1964) et Watson (1964) pour la régression à noyau, Fan et Gijbels (1996) pour les polynômes locaux, Robinson (1988) pour le modèle partiellement linéaire, Silverman (1986) et Li et Racine (2007) pour les méthodes à noyau.
