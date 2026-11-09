## Définition

Notations : $y_i$ est la variable expliquée de l'observation $i$, $x_i$ le vecteur de ses $K$ régresseurs, $\beta$ le vecteur des coefficients et $\theta$ l'ensemble de tous les paramètres du modèle (les coefficients, mais aussi les seuils, $\sigma$, $\rho$, etc.). $\Phi$ et $\phi$ sont la fonction de répartition et la densité de la loi normale centrée réduite, et $\Lambda(a) = 1/(1 + e^{-a})$ la fonction de répartition de la loi logistique.

### Maximum de vraisemblance

- **Vraisemblance** : probabilité (ou densité) d'observer les données telles qu'elles sont, vue comme une fonction des paramètres, $L(\theta) = \prod_i f(y_i \mid x_i ; \theta)$.
- **Log-vraisemblance** : son logarithme, qui transforme le produit en somme et est plus stable numériquement,
  $$\ell(\theta) = \sum_{i=1}^n \ln f(y_i \mid x_i ; \theta)$$
- **Maximum de vraisemblance** (MV) : on choisit les paramètres qui rendent les données observées les plus probables, $\hat\theta = \arg\max_\theta \ell(\theta)$. Pour ajouter un modèle dans la librairie, il suffit de donner $\ln f(y_i \mid x_i ; \theta)$ pour chaque observation.
- **Score** : gradient de la log-vraisemblance, $s_i(\theta) = \partial \ln f(y_i \mid x_i ; \theta) / \partial \theta$. Au maximum, $\sum_i s_i(\hat\theta) = 0$.
- **Hessien** : matrice des dérivées secondes, $H(\theta) = \sum_i \partial^2 \ln f(y_i \mid x_i ; \theta) / \partial\theta\,\partial\theta^\top$. Au maximum elle est définie négative.
- **Dérivées analytiques / numériques** : analytiques si on les calcule à la main (logit, probit), numériques si on les approxime par différences finies. Pour la composante $j$ du gradient, avec un petit pas $h$ :
  $$\frac{\partial \ell}{\partial \theta_j} \approx \frac{\ell(\theta + h e_j) - \ell(\theta - h e_j)}{2h}$$
  Cela demande $2p$ évaluations de $\ell$ pour $p$ paramètres.
- **Newton-Raphson** : algorithme itératif qui utilise le score et le hessien,
  $$\theta_{k+1} = \theta_k - \lambda_k \, H(\theta_k)^{-1} s(\theta_k)$$
  où $s = \sum_i s_i$. Converge en très peu d'itérations près de l'optimum mais demande le hessien.
- **Recherche de pas** : choix du coefficient $\lambda_k \in (0, 1]$ de façon à ce que $\ell$ augmente bien d'une itération à l'autre (on divise $\lambda_k$ par 2 tant que ce n'est pas le cas).
- **BFGS** : méthode quasi-Newton qui n'a besoin que du gradient. Elle construit pas à pas une approximation $B_k$ de l'inverse du hessien, avec $\delta = \theta_{k+1} - \theta_k$, $\gamma = g_{k+1} - g_k$ (variation du gradient) et $\rho = 1/(\gamma^\top \delta)$ :
  $$B_{k+1} = (I - \rho\,\delta\gamma^\top) \, B_k \, (I - \rho\,\gamma\delta^\top) + \rho\,\delta\delta^\top$$
  Elle est plus robuste que Newton loin de l'optimum, c'est pourquoi elle sert de première phase pour les modèles à dérivées numériques.
- **Matrice d'information** : $\mathcal I(\theta) = -E[H(\theta)]$. Son inverse est la variance asymptotique du MV si le modèle est bien spécifié.

### Variances du MV

Notons $\hat H = H(\hat\theta)$ et $\hat s_i = s_i(\hat\theta)$.

- **Variance par le hessien** : $\widehat{\text{Var}}(\hat\theta) = (-\hat H)^{-1}$.
- **OPG** (*outer product of gradients*) : utilise le produit externe des scores à la place du hessien, $\widehat{\text{Var}}(\hat\theta) = \big(\sum_i \hat s_i \hat s_i^\top\big)^{-1}$. Vaut le hessien si le modèle est bien spécifié.
- **Variance robuste (sandwich)** : reste valable même si la forme de la loi est un peu fausse,
  $$\widehat{\text{Var}}(\hat\theta) = (-\hat H)^{-1} \Big( \sum_i \hat s_i \hat s_i^\top \Big) (-\hat H)^{-1}$$
- **Variance clusterisée** : même sandwich, mais on somme d'abord les scores à l'intérieur de chaque groupe $g$ (erreurs corrélées dans un groupe, indépendantes entre groupes),
  $$\widehat{\text{Var}}(\hat\theta) = (-\hat H)^{-1} \Big( \sum_{g} \big(\sum_{i \in g} \hat s_i\big) \big(\sum_{i \in g} \hat s_i\big)^\top \Big) (-\hat H)^{-1}$$
- **Écart-type et p-value** : l'écart-type est la racine de l'élément diagonal de la variance. La statistique $z_j = \hat\theta_j / \text{se}(\hat\theta_j)$ suit approximativement une loi normale centrée réduite sous $H_0 : \theta_j = 0$, d'où des p-values normales.

### Variables dépendantes binaires

- **Variable latente** : variable non observée $y_i^* = x_i^\top\beta + \varepsilon_i$ dont on n'observe qu'une transformation. Pour un modèle binaire, $y_i = 1$ si $y_i^* > 0$ et $0$ sinon.
- **Logit** : $\varepsilon$ suit une loi logistique, donc $P(y_i = 1 \mid x_i) = \Lambda(x_i^\top\beta)$.
- **Probit** : $\varepsilon$ suit une loi normale, donc $P(y_i = 1 \mid x_i) = \Phi(x_i^\top\beta)$.
- **Log-vraisemblance binaire** (avec $F = \Lambda$ ou $\Phi$) :
  $$\ell(\beta) = \sum_i \big[ y_i \ln F(x_i^\top\beta) + (1 - y_i) \ln\big(1 - F(x_i^\top\beta)\big) \big]$$
  Pour le logit, le score est $\sum_i (y_i - \Lambda_i) x_i$ et le hessien est $-\sum_i \Lambda_i (1 - \Lambda_i) x_i x_i^\top$ avec $\Lambda_i = \Lambda(x_i^\top\beta)$.
- **Pseudo R2 (McFadden)** : $1 - \ell(\hat\theta) / \ell_0$, où $\ell_0$ est la log-vraisemblance du modèle avec seulement une constante. Il n'a pas la même interprétation que le $R^2$ des MCO.

### Choix ordonnés et multinomiaux

- **Modèle ordonné** (logit ou probit) : la variable prend des valeurs $1, \dots, J$ qui ont un ordre naturel. On observe $y_i = j$ si $y_i^*$ tombe entre deux **seuils** $\kappa_{j-1} < y_i^* \le \kappa_j$, avec $\kappa_0 = -\infty$ et $\kappa_J = +\infty$ :
  $$P(y_i = j \mid x_i) = F(\kappa_j - x_i^\top\beta) - F(\kappa_{j-1} - x_i^\top\beta)$$
  Les seuils sont des paramètres à estimer (d'où "seuils estimés avec leur écart-type"), et il n'y a pas de constante dans $x_i$, car elle ne serait pas séparable des seuils.
- **Logit multinomial** : la variable prend $J$ valeurs sans ordre. On fixe une catégorie de référence ($\beta_0 = 0$) et
  $$P(y_i = j \mid x_i) = \frac{\exp(x_i^\top\beta_j)}{\sum_{l=0}^{J-1} \exp(x_i^\top\beta_l)}$$
  Il y a donc $p = (J-1)K$ paramètres, ce qui explique le coût qui augmente avec $J$.

### Variables dépendantes limitées

- **Censure** : on observe bien $x_i$, mais la variable $y_i^*$ est coupée à un seuil $c$ (par exemple on ne voit que $\max(y_i^*, 0)$). Les valeurs censurées sont dans l'échantillon, au seuil.
- **Tobit** : modèle linéaire sur $y^*$ avec une erreur normale, $y_i^* = x_i^\top\beta + \varepsilon_i$, $\varepsilon_i \sim N(0, \sigma^2)$, observé avec censure à gauche en $c$ ($y_i = \max(y_i^*, c)$). Les observations non censurées contribuent par leur densité, les observations censurées par la probabilité d'être censurées :
  $$\ell_i = \mathbb 1(y_i > c) \Big[ \ln\phi\Big(\frac{y_i - x_i^\top\beta}{\sigma}\Big) - \ln\sigma \Big] + \mathbb 1(y_i = c) \ln\Phi\Big(\frac{c - x_i^\top\beta}{\sigma}\Big)$$
  La censure à droite est symétrique, avec $\ln\big(1 - \Phi(\cdot)\big)$. Les MCO sont biaisés vers zéro dans ce cas.
- **Sélection d'échantillon** : on n'observe $y_i$ que pour une partie non aléatoire de l'échantillon, selon une règle qui dépend de variables liées à l'erreur. Les MCO sur la partie observée sont alors biaisés.
- **Modèle de Heckman** : on combine une équation de sélection et une équation d'intérêt,
  $$s_i^* = z_i^\top\gamma + u_i, \quad s_i = \mathbb 1(s_i^* > 0), \qquad y_i = x_i^\top\beta + \varepsilon_i \ \text{observé si } s_i = 1$$
  avec $(u_i, \varepsilon_i)$ normale bivariée, $\text{Var}(u) = 1$, $\text{Var}(\varepsilon) = \sigma^2$ et corrélation $\rho$. Si $\rho \neq 0$, la sélection est endogène et
  $$E[y_i \mid x_i, s_i = 1] = x_i^\top\beta + \rho\sigma \, \lambda(z_i^\top\gamma), \qquad \lambda(a) = \frac{\phi(a)}{\Phi(a)}$$
  où $\lambda$ est l'**inverse du ratio de Mills**. Oublier ce terme fait un biais de variable omise dans les MCO. La log-vraisemblance complète est
  $$\ell = \sum_{s_i = 0} \ln\Phi(-z_i^\top\gamma) + \sum_{s_i = 1} \Big[ \ln\phi\Big(\frac{y_i - x_i^\top\beta}{\sigma}\Big) - \ln\sigma + \ln\Phi\Big( \frac{z_i^\top\gamma + \rho (y_i - x_i^\top\beta)/\sigma}{\sqrt{1 - \rho^2}} \Big) \Big]$$
- **Restriction d'exclusion** : pour bien identifier le modèle, $Z$ doit contenir au moins une variable qui n'est pas dans $X$ (elle explique la sélection mais pas $y$ directement). Sans elle, l'identification repose seulement sur la forme de la loi normale.
- **Heckman en deux étapes** : variante (non incluse) où on estime d'abord $\hat\gamma$ par un probit, puis on ajoute $\hat\lambda(z_i^\top\hat\gamma)$ comme régresseur dans un MCO.

### Effets marginaux

- **Effet marginal** : variation de la quantité d'intérêt (une probabilité, une espérance) quand $x_k$ augmente d'une unité. Dans un modèle non linéaire il dépend de $x$ et n'est pas égal au coefficient $\beta_k$. Pour un modèle binaire,
  $$\frac{\partial P(y_i = 1 \mid x_i)}{\partial x_{ik}} = f(x_i^\top\beta) \, \beta_k$$
  avec $f = \Lambda(1 - \Lambda)$ pour le logit et $f = \phi$ pour le probit.
- **Effet marginal moyen** (*average marginal effect*) : on calcule l'effet pour chaque observation, puis on fait la moyenne,
  $$\text{EMM}_k = \frac{1}{n} \sum_i f(x_i^\top\hat\beta) \, \hat\beta_k$$
- **Effet marginal en la moyenne** : on calcule l'effet une seule fois, au point $x = \bar x$ (la moyenne des régresseurs), $f(\bar x^\top\hat\beta) \, \hat\beta_k$.
- **Cas du Tobit** : l'effet de $x_k$ sur la variable latente est $\beta_k$, mais sur la variable observée (avec censure en 0) il vaut $\beta_k \, \Phi(x^\top\beta/\sigma)$.
- **Delta method** : permet d'obtenir l'écart-type d'une fonction des paramètres $g(\hat\theta)$ (ici l'effet marginal) à partir de la variance de $\hat\theta$,
  $$\widehat{\text{Var}}\big(g(\hat\theta)\big) \approx \nabla g(\hat\theta)^\top \, \widehat{\text{Var}}(\hat\theta) \, \nabla g(\hat\theta)$$
  Le gradient $\nabla g$ est calculé numériquement dans la librairie.

### Tests

- **Test de Wald** : teste $q$ restrictions linéaires $R\theta = r$ en estimant seulement le modèle non contraint,
  $$W = (R\hat\theta - r)^\top \big[ R \, \widehat{\text{Var}}(\hat\theta) \, R^\top \big]^{-1} (R\hat\theta - r) \sim \chi^2(q)$$
- **Test du rapport de vraisemblance** (LR) : compare le modèle libre et le modèle contraint par les restrictions, en estimant les deux. Avec $\ell_U$ et $\ell_R$ leurs log-vraisemblances,
  $$LR = 2(\ell_U - \ell_R) \sim \chi^2(q)$$
  Une valeur élevée indique que les restrictions dégradent fortement l'ajustement.

### Vocabulaire de la complexité

- **$O(\cdot)$** : ordre de grandeur du nombre d'opérations (temps) ou de la place utilisée (mémoire), en ignorant les constantes.
- **$n, K, J, p, I$** : nombre d'observations, de régresseurs, de catégories, de paramètres estimés et d'itérations de l'optimiseur.
