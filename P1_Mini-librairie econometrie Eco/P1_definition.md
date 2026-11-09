## Définition

Notations : $y$ est le vecteur des $n$ observations de la variable expliquée, $X$ la matrice $n \times k$ des régresseurs, $Z$ la matrice $n \times m$ des instruments et des variables exogènes, $\beta$ le vecteur des coefficients et $\varepsilon$ le terme d'erreur. $e = y - X\hat\beta$ désigne les résidus et $\hat\beta$ un estimateur de $\beta$.

### Modèle et problèmes à traiter

- **Modèle linéaire** : $y = X\beta + \varepsilon$.
- **Exogénéité** : les régresseurs ne sont pas corrélés avec l'erreur, $E[\varepsilon \mid X] = 0$. C'est la condition pour que les MCO soient sans biais.
- **Endogénéité** : un régresseur est corrélé avec l'erreur, $\text{Cov}(x, \varepsilon) \neq 0$ (variable omise, erreur de mesure, simultanéité). Les MCO sont alors biaisés, même avec beaucoup de données.
- **Homoscédasticité** : la variance de l'erreur est constante, $\text{Var}(\varepsilon_i \mid x_i) = \sigma^2$.
- **Hétéroscédasticité** : la variance de l'erreur change selon les observations, $\text{Var}(\varepsilon_i \mid x_i) = \sigma_i^2$. Les coefficients des MCO restent sans biais mais les écarts-types classiques sont faux.
- **Hétéroscédasticité multiplicative** : forme particulière utilisée pour le MCGF, $\sigma_i^2 = \exp(a + b x_i)$.
- **Variable instrumentale (instrument)** : variable $z$ qui sert à corriger l'endogénéité. Elle doit être corrélée avec le régresseur endogène (pertinence, $\text{Cov}(z, x) \neq 0$) et pas avec l'erreur (exogénéité, $\text{Cov}(z, \varepsilon) = 0$).
- **Sur-identification** : il y a plus d'instruments que de régresseurs endogènes ($m > k$). C'est ce qui permet de tester la validité des instruments.

### Estimateurs

- **MCO** (moindres carrés ordinaires, *OLS* en anglais) : minimise la somme des carrés des résidus.
  $$\hat\beta_{MCO} = (X^\top X)^{-1} X^\top y$$
- **MCG** (moindres carrés généralisés, *GLS*) : MCO pondérés quand on connaît la matrice de variance des erreurs $\Omega$. Les observations les plus bruitées pèsent moins.
  $$\hat\beta_{MCG} = (X^\top \Omega^{-1} X)^{-1} X^\top \Omega^{-1} y \qquad \text{Var}(\hat\beta_{MCG}) = (X^\top \Omega^{-1} X)^{-1}$$
- **MCGF** (moindres carrés généralisés faisables, *FGLS*) : comme le MCG mais $\Omega$ est estimée. Ici on régresse $\ln(e_i^2)$ sur les régresseurs pour estimer $\hat\sigma_i^2 = \exp(\hat a + \hat b x_i)$, puis on refait une régression pondérée par $1/\hat\sigma_i^2$ (deux passes de MCO).
- **2SLS** (doubles moindres carrés, *two-stage least squares*, aussi noté MC2E) : première étape, on régresse $X$ sur $Z$ pour obtenir la partie de $X$ expliquée par les instruments. Deuxième étape, on régresse $y$ sur cette partie.
  $$P_Z = Z(Z^\top Z)^{-1} Z^\top \qquad \hat\beta_{2SLS} = (X^\top P_Z X)^{-1} X^\top P_Z y$$
- **GMM** (méthode des moments généralisés, *generalized method of moments*) : on part des conditions de moments $E[z_i \varepsilon_i] = 0$, dont la version empirique est
  $$g(\beta) = \frac{1}{n} Z^\top (y - X\beta)$$
  et on choisit $\beta$ pour rendre $g(\beta)$ proche de zéro, au sens d'une matrice de pondération $W$ :
  $$\hat\beta_{GMM} = \arg\min_\beta \; g(\beta)^\top W g(\beta) = (X^\top Z W Z^\top X)^{-1} X^\top Z W Z^\top y$$
- **GMM en deux étapes** : première étape avec $W_1 = (Z^\top Z / n)^{-1}$ (ce qui redonne le 2SLS), puis seconde étape avec le poids optimal robuste à l'hétéroscédasticité
  $$W_2 = \hat S^{-1}, \qquad \hat S = \frac{1}{n} \sum_{i=1}^n e_i^2 \, z_i z_i^\top$$
- **Matrice de pondération** ($W$) : elle dit quelles conditions de moments comptent le plus dans le GMM.

### Écarts-types

Tous les écarts-types ci-dessous sont la racine des éléments diagonaux d'une matrice de variance de $\hat\beta$.

- **Écart-type classique** : valable sous homoscédasticité.
  $$\widehat{\text{Var}}(\hat\beta) = s^2 (X^\top X)^{-1}, \qquad s^2 = \frac{e^\top e}{n - k}$$
- **Écart-type robuste** : reste valable sous hétéroscédasticité. Il a la forme dite "sandwich" (deux fois $(X^\top X)^{-1}$ autour d'une matrice du milieu) :
  $$(X^\top X)^{-1} \Big( \sum_{i=1}^n \omega_i \, e_i^2 \, x_i x_i^\top \Big) (X^\top X)^{-1}$$
- **White, HC0 à HC3** (*heteroskedasticity-consistent*) : quatre versions du sandwich, qui ne diffèrent que par le poids $\omega_i$. On note $h_i = x_i^\top (X^\top X)^{-1} x_i$ le levier de l'observation $i$.

  | Version | $\omega_i$ |
  |---|---|
  | HC0 | $1$ |
  | HC1 | $n / (n - k)$ |
  | HC2 | $1 / (1 - h_i)$ |
  | HC3 | $1 / (1 - h_i)^2$ |

  HC2 et HC3 corrigent le fait que les observations à fort levier ont des résidus trop petits. HC3 est le plus prudent en petit échantillon.
- **Newey-West, HAC** (*heteroskedasticity and autocorrelation consistent*) : robuste aussi à l'autocorrélation des erreurs jusqu'à $L$ retards. La matrice du milieu devient
  $$S = \Gamma_0 + \sum_{l=1}^{L} \Big(1 - \frac{l}{L+1}\Big) (\Gamma_l + \Gamma_l^\top), \qquad \Gamma_l = \sum_{t=l+1}^{n} e_t e_{t-l} \, x_t x_{t-l}^\top$$
  et la variance vaut $(X^\top X)^{-1} S (X^\top X)^{-1}$. Les poids $1 - l/(L+1)$ sont ceux du noyau de Bartlett.
- **Écart-type clusterisé** : robuste à une corrélation des erreurs à l'intérieur de groupes (clusters) $g = 1, \dots, G$, avec indépendance entre groupes. Avec $X_g$ et $e_g$ les lignes de $X$ et les résidus du groupe $g$ :
  $$(X^\top X)^{-1} \Big( \sum_{g=1}^{G} X_g^\top e_g e_g^\top X_g \Big) (X^\top X)^{-1}$$
  (souvent multiplié par une correction de petit échantillon qui dépend de $G$, $n$ et $k$).

### Tests

- **Statistique de Student** : pour un coefficient, $t_j = \hat\beta_j / \text{se}(\hat\beta_j)$. La **p-value** est la probabilité, sous l'hypothèse nulle $\beta_j = 0$, d'obtenir une statistique au moins aussi extrême que celle observée. Ici elle est calculée avec une loi de Student à $n - k$ degrés de liberté (*degrés de liberté* : nombre d'observations moins nombre de paramètres estimés).
- **Test de Wald** : teste $q$ restrictions linéaires $R\beta = r$ en même temps.
  $$W = (R\hat\beta - r)^\top \big[ R \, \widehat{\text{Var}}(\hat\beta) \, R^\top \big]^{-1} (R\hat\beta - r)$$
  Sous l'hypothèse nulle, $W$ suit asymptotiquement une loi $\chi^2(q)$. En petit échantillon on utilise parfois $W/q$ avec une loi de Fisher.
- **Test de Breusch-Pagan** : teste l'homoscédasticité. On régresse les résidus au carré $e_i^2$ sur les régresseurs et on garde le $R^2$ de cette régression auxiliaire. La statistique est
  $$LM = n R^2_{aux} \sim \chi^2(p)$$
  où $p$ est le nombre de régresseurs de la régression auxiliaire hors constante. Une p-value faible indique de l'hétéroscédasticité.
- **Test de Hausman** : compare un estimateur robuste à l'endogénéité (2SLS) à un estimateur efficace si les régresseurs sont exogènes (MCO). S'ils sont proches, les MCO sont acceptables ; sinon il y a endogénéité.
  $$H = (\hat\beta_{2SLS} - \hat\beta_{MCO})^\top \big[ \widehat{\text{Var}}(\hat\beta_{2SLS}) - \widehat{\text{Var}}(\hat\beta_{MCO}) \big]^{-1} (\hat\beta_{2SLS} - \hat\beta_{MCO}) \sim \chi^2(r)$$
  où $r$ est le nombre de coefficients comparés (l'argument `indices`).
- **J de Hansen** : test de sur-identification après un GMM. Il vérifie que les moments en trop sont bien proches de zéro, c'est-à-dire que les instruments sont valides.
  $$J = n \, g(\hat\beta)^\top \hat S^{-1} g(\hat\beta) \sim \chi^2(m - k)$$
  Une p-value faible suggère qu'au moins un instrument n'est pas exogène.

### Vocabulaire de la complexité

- **$O(\cdot)$** : ordre de grandeur du nombre d'opérations (temps) ou de la place utilisée (mémoire) quand les tailles augmentent, en ignorant les constantes. Par exemple $O(nk^2)$ veut dire que le coût double quand $n$ double.
- **$n, k, m, L, G, q$** : nombre d'observations, de régresseurs, de colonnes de $Z$, de retards du HAC, de clusters et de restrictions testées par Wald.
