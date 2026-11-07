# mini-montecarlo

Un petit banc d'essai Monte Carlo en numpy pour vérifier les résultats classiques d'économétrie asymptotique, dans l'esprit des chapitres 4, 5, 8, 9 et 15 de *Econometric Analysis* de Greene : consistance, normalité asymptotique, taille et puissance des tests, hétéroscédasticité, instruments faibles et bootstrap. Les estimateurs sont écrits pour traiter des milliers de replications d'un coup, sans boucle Python.

## Ce qu'il y a dedans

- **Biais et consistance** : MCO exogène, MCO endogène et 2SLS quand n grandit, avec la vitesse en racine de n
- **Normalité asymptotique** de la statistique t robuste avec des erreurs très asymétriques
- **Taille des tests** : variance classique, White (HC0 à HC3), wild bootstrap et bootstrap par paires (percentile-t)
- **Puissance** des tests en fonction de l'effet
- **Instruments faibles** : biais et taille des 2SLS et du LIML, test d'Anderson-Rubin
- Des générateurs de données avec hétéroscédasticité, régresseurs asymétriques (forts leviers) et erreurs non normales

## Installation

```
pip install -r requirements.txt
```

Seuls numpy et scipy servent à faire tourner le code. statsmodels, linearmodels et pytest ne sont là que pour les tests.

## Exemple rapide

```python
import montecarlo as mc

print(mc.taille(n=50, hetero=0.5, regresseurs="asymetrique"))
print(mc.tableau(*mc.instruments_faibles(mu2s=(1, 10, 100))))
```

`python exemple.py` lance toutes les expériences en une vingtaine de secondes. Ce qu'on y voit :

- **Consistance** : le MCO endogène reste biaisé de 0.405 même à n = 1600 (la valeur théorique est 0.6 / 1.48), alors que le biais des 2SLS tombe à 0.000. Le produit racine de n fois l'écart-type se stabilise vers 1.46 pour les 2SLS.
- **Taille avec hétéroscédasticité et forts leviers** (n = 50, niveau nominal 5 %) : la variance classique rejette 50 % du temps à tort, HC0 et HC1 environ 25 %, HC2 15 %, HC3 7 %, le wild bootstrap 7 % et le bootstrap par paires 5 %. Avec des erreurs homoscédastiques, tout est proche de 5 %.
- **Instruments faibles** (5 instruments, n = 200) : quand la première étape est très faible (F moyen de 1.2), le test t des 2SLS rejette 62 % du temps à tort et le LIML 26 %, alors que le test d'Anderson-Rubin reste à 5 % partout. Le biais médian des 2SLS (0.66) reste proche de celui du MCO (0.80), contrairement au LIML.

## Organisation

```
montecarlo/
    simulation.py    générateurs de données
    estimateurs.py   MCO et IV sur de nombreuses replications à la fois
    bootstrap.py     wild bootstrap et bootstrap par paires
    experiences.py   les cinq expériences et la mise en forme des tableaux
tests/
    test_validation.py
exemple.py
benchmark.py
```

## Complexité

Notations : R replications, n observations, k régresseurs, B tirages bootstrap, m instruments.

| Méthode | Temps | Mémoire |
|---|---|---|
| MCO sur R échantillons | O(R(nk² + k³)) | O(Rnk) |
| Écarts-types HC0 à HC3 | O(Rnk) | O(Rnk) |
| 2SLS, LIML, Anderson-Rubin | O(R(nm² + m³)) | O(Rnm) |
| Wild bootstrap | O(RBnk) | O(CBn) |
| Bootstrap par paires | O(RB(nk² + k³)) | O(CBnk) |

C est la taille d'un bloc de replications traité à la fois. Elle est choisie pour garder les tableaux sous quelques millions d'éléments, donc la mémoire du bootstrap ne dépend pas de R. Le wild bootstrap est bien moins cher que le bootstrap par paires parce que X ne change pas : la matrice (X'X)⁻¹X' est calculée une fois par replication, et chaque tirage ne coûte plus qu'un produit matrice-vecteur. Sur 2000 replications de 199 tirages (n = 50), le wild bootstrap prend environ 0.4 s et le bootstrap par paires environ 1.8 s.

## Validation

```
pytest
```

Les 13 tests couvrent deux niveaux :

- **Exactitude des calculs** : les MCO par lots donnent les mêmes coefficients et écarts-types que `statsmodels` (classique, HC0 à HC3). Les 2SLS et le LIML donnent les mêmes coefficients et écarts-types que `linearmodels`. Les deux bootstraps sont comparés à une boucle naïve qui refait chaque régression avec `statsmodels`, tirage par tirage, avec les mêmes poids ou indices.
- **Résultats théoriques** : taille proche de 5 % sans hétéroscédasticité, test classique faussé avec hétéroscédasticité, ordre HC0 > HC2 > HC3 sous forts leviers, puissance croissante, biais du MCO endogène égal à 0.6 / 1.48 à n grand, biais des 2SLS qui disparaît, taille de l'Anderson-Rubin à 5 % même avec des instruments très faibles.

`python benchmark.py` compare les versions vectorisées à des boucles simples en numpy. Le gain est modeste : environ ×3 pour 5000 régressions avec sandwich HC3, et environ ×4 pour le wild bootstrap. Les matrices sont petites, donc une boucle Python n'est déjà pas très lente ici.

## Limites

- Les variables instrumentales n'ont qu'un régresseur endogène, sans constante ni contrôles exogènes. C'est ce qui permet de tout calculer avec six produits scalaires.
- Pour l'écart-type du LIML, j'utilise la variance de la classe k avec σ² estimé en divisant par n, comme `linearmodels` en version non ajustée. Ce n'est pas la correction de Bekker.
- Le bootstrap par paires est un peu conservateur : à n = 200 avec forts leviers, il rejette 2.5 % pour un niveau nominal de 5 %, un écart bien supérieur au bruit de simulation (environ 0.5 point avec 2000 replications).
- Le test d'Anderson-Rubin suppose des erreurs normales pour que sa loi soit exactement une loi F.
- Dans l'expérience de puissance, le test classique rejette déjà 13.5 % sous H0 à cause de l'hétéroscédasticité. Sa puissance brute est donc meilleure en apparence uniquement.
- Les fonctions du bootstrap sont écrites pour un coefficient à la fois, avec un modèle de régression linéaire.

## Références

Greene, *Econometric Analysis* (8e édition) : chapitres 4 et 5 (propriétés en échantillon fini et asymptotiques), 8 (variables instrumentales et instruments faibles), 9 (hétéroscédasticité) et 15 (simulation et bootstrap). MacKinnon et White (1985) pour HC2 et HC3, Wu (1986) et Davidson et Flachaire (2008) pour le wild bootstrap, Anderson et Rubin (1949) pour le test du même nom.
