## Définition

Notations : $y_i$ est la variable expliquée de l'observation $i$, $x_i$ le vecteur de ses $k$ régresseurs (avec une constante) et $n$ le nombre d'observations. Dans la partie non paramétrique, $x$ est une seule variable. $K$ désigne un noyau et $h$ une fenêtre.

### Régression quantile

- **Quantile** : le quantile d'ordre $\tau \in (0,1)$ d'une variable est la valeur sous laquelle se trouve une proportion $\tau$ des observations. $\tau = 0.5$ est la médiane, $\tau = 0.9$ le neuvième décile.
- **Quantile conditionnel** : quantile de $y$ sachant $x$. La régression quantile le modélise linéairement,
  $$Q_\tau(y \mid x) = x^\top \beta(\tau)$$
  Les coefficients dépendent de $\tau$ : on peut voir comment l'effet de $x$ change entre le bas et le haut de la distribution de $y$, ce que les MCO (qui ne décrivent que la moyenne) ne montrent pas.
- **Fonction de perte quantile** (*check function*) : pénalise asymétriquement les erreurs positives et négatives,
  $$\rho_\tau(u) = u \, \big(\tau - \mathbb 1(u < 0)\big), \qquad \hat\beta(\tau) = \arg\min_\beta \sum_{i=1}^n \rho_\tau(y_i - x_i^\top\beta)$$
  Pour $\tau = 0.5$ on retrouve la minimisation de la somme des écarts absolus (régression médiane).
- **Programmation linéaire** : le problème précédent s'écrit comme un programme linéaire, ce qui donne une solution exacte,
  $$\min_{\beta, u^+, u^-} \; \tau \, \mathbf 1^\top u^+ + (1-\tau) \, \mathbf 1^\top u^- \quad \text{sous} \quad X\beta + u^+ - u^- = y, \; u^+, u^- \ge 0$$
  où $u^+$ et $u^-$ sont les parties positive et négative des résidus. À l'optimum, exactement $k$ résidus sont nuls : ce sont les **points d'interpolation**, par lesquels passe l'hyperplan estimé.
- **Dual** : le programme équivalent écrit du côté des contraintes,
  $$\max_{a} \; y^\top a \quad \text{sous} \quad X^\top a = (1-\tau) X^\top \mathbf 1, \quad a \in [0, 1]^n$$
  Il n'a que $k$ contraintes d'égalité au lieu de $n$, d'où un calcul plus rapide.
- **Points intérieurs, HiGHS** : le **solveur** HiGHS est un logiciel libre de programmation linéaire. La méthode des **points intérieurs** avance à l'intérieur de la région admissible plutôt que le long de ses sommets (simplexe), ce qui est plus efficace quand $n$ est grand.
- **Moindres carrés repondérés itérés** (IRLS, utilisés par `QuantReg` de statsmodels) : on approche la perte quantile par des moindres carrés pondérés qu'on réévalue à chaque itération. Plus rapide mais approché.
- **Pseudo R2 de Koenker et Machado** : compare la perte du modèle à celle d'un modèle avec seulement une constante,
  $$R^1(\tau) = 1 - \frac{\sum_i \rho_\tau(y_i - x_i^\top\hat\beta(\tau))}{\sum_i \rho_\tau(y_i - \hat q_\tau)}$$
  où $\hat q_\tau$ est le quantile empirique de $y$. Il se lit comme un $R^2$, mais pour le quantile $\tau$.
- **Courbe d'Engel** : relation entre la part ou le montant des dépenses d'un poste (ici l'alimentation) et le revenu du ménage. Dans l'exemple, la pente varie selon le quantile.
- **Modèle à position et échelle** : $y_i = x_i^\top\beta + (x_i^\top\gamma)\,\varepsilon_i$. Le quantile conditionnel vaut $x_i^\top(\beta + \gamma \, q_\tau)$ où $q_\tau$ est le quantile de $\varepsilon$ : les pentes varient avec $\tau$ dès que l'échelle dépend de $x$ (hétéroscédasticité). C'est le cas simulé pour vérifier les valeurs théoriques.

### Variances de la régression quantile

- **Variance iid** : suppose des erreurs de même loi, avec $\hat f(0)$ la densité estimée des résidus en 0,
  $$\widehat{\text{Var}}(\hat\beta) = \frac{\tau(1-\tau)}{\hat f(0)^2} (X^\top X)^{-1}$$
- **Sandwich de Powell** : robuste à l'hétéroscédasticité. Avec $\hat f_i$ la densité des résidus en 0 estimée par un noyau, $\hat f_i = K(e_i / h)/h$ pour la fenêtre $h$,
  $$\widehat{\text{Var}}(\hat\beta) = \tau(1-\tau) \, \hat H^{-1} (X^\top X) \hat H^{-1}, \qquad \hat H = \sum_i \hat f_i \, x_i x_i^\top$$
- **Bootstrap par paires** : on tire avec remise des couples $(y_i, x_i)$, on réestime $\hat\beta(\tau)$ sur chaque tirage ($B$ fois), et on prend la variance empirique des $B$ estimations.

### Plusieurs quantiles

- **Croisement de quantiles** : comme chaque quantile est estimé séparément, il arrive que $\hat Q_{0.9}(y \mid x) < \hat Q_{0.5}(y \mid x)$ pour certains $x$, ce qui est impossible pour de vrais quantiles.
- **Réordonnancement** (*rearrangement*, Chernozhukov, Fernandez-Val et Galichon, 2010) : pour chaque $x$, on trie simplement les valeurs prévues dans l'ordre croissant des $\tau$. Cela supprime les croisements et ne dégrade pas l'estimation.
- **Test d'égalité des pentes** : teste $H_0 : \beta_{-1}(\tau_1) = \beta_{-1}(\tau_2)$, c'est-à-dire que les pentes (hors constante) sont les mêmes à deux quantiles. C'est un test de Wald,
  $$W = (\hat\beta_{\tau_1} - \hat\beta_{\tau_2})^\top \big[ V_{11} + V_{22} - V_{12} - V_{21} \big]^{-1} (\hat\beta_{\tau_1} - \hat\beta_{\tau_2}) \sim \chi^2(k - 1)$$
  avec la covariance conjointe obtenue par bootstrap, $V_{12}$ étant la covariance entre les deux estimateurs. Le rejeter veut dire que l'effet de $x$ n'est pas le même en bas et en haut de la distribution.

### Estimation à noyau

- **Non paramétrique** : on n'impose pas de forme fonctionnelle (linéaire, quadratique...), c'est la donnée qui détermine la courbe. Le prix à payer est une convergence plus lente (de l'ordre de $n^{-2/5}$, contre $n^{-1/2}$ pour un modèle paramétrique).
- **Noyau** ($K$) : fonction positive d'intégrale 1, symétrique autour de 0, qui donne un poids décroissant aux observations en fonction de leur distance au point où on estime. Exemples : gaussien $K(u) = \phi(u)$, d'Epanechnikov $K(u) = \tfrac{3}{4}(1 - u^2)\,\mathbb 1(|u| \le 1)$, triangulaire $K(u) = (1 - |u|)\,\mathbb 1(|u| \le 1)$. Un noyau est à **support compact** s'il est nul hors de $[-1, 1]$. Dans ce cas, un point sans aucune observation à moins de $h$ n'a pas d'estimation (`nan`).
- **Fenêtre** ($h$, *bandwidth*) : largeur du voisinage utilisé. Une fenêtre petite donne une courbe très irrégulière (peu de biais, beaucoup de variance), une grande donne une courbe trop lisse (peu de variance, beaucoup de biais). C'est le paramètre le plus important.
- **Densité à noyau** : estimation de la densité de $x$ sans supposer de loi,
  $$\hat f(x) = \frac{1}{nh} \sum_{i=1}^n K\Big(\frac{x - x_i}{h}\Big)$$
- **Règles de Silverman et de Scott** : fenêtres de référence pour la densité, calculées sans optimisation. Dans `statsmodels`, avec $A = \min(\hat\sigma, \text{IQR}/1.349)$ où IQR est l'écart interquartile,
  $$h_{Silverman} = 0.9 \, A \, n^{-1/5}, \qquad h_{Scott} = 1.059 \, A \, n^{-1/5}$$
- **Nadaraya-Watson** : estime $m(x) = E[y \mid x]$ par une moyenne pondérée locale des $y_i$,
  $$\hat m(x) = \frac{\sum_i K\big((x_i - x)/h\big) \, y_i}{\sum_i K\big((x_i - x)/h\big)}$$
- **Polynôme local de degré $d$** : au lieu d'une moyenne, on ajuste un polynôme par moindres carrés pondérés autour de $x$ et on garde sa valeur en $x$,
  $$\hat m(x) = \hat b_0, \qquad \hat b = \arg\min_b \sum_i K\Big(\frac{x_i - x}{h}\Big) \Big( y_i - \sum_{j=0}^{d} b_j (x_i - x)^j \Big)^2$$
  Le degré 0 est Nadaraya-Watson, le degré 1 est le **linéaire local**. L'estimateur est linéaire en $y$ : $\hat m(x_i)$ s'écrit $\sum_j L_{ij} y_j$, avec $L$ la matrice de lissage.
- **Effet de bord** : près des extrémités de l'échantillon, le voisinage est asymétrique. Nadaraya-Watson est alors biaisé (d'ordre $h \, m'(x)$), le linéaire local corrige ce biais. D'où l'écart observé en $x = 0$ dans l'exemple.
- **Validation croisée leave-one-out** : on choisit $h$ qui minimise l'erreur de prévision quand chaque observation est prédite sans elle-même,
  $$CV(h) = \frac{1}{n} \sum_{i=1}^n \Big( \frac{y_i - \hat m_h(x_i)}{1 - L_{ii}} \Big)^2$$
  La fraction $e_i / (1 - L_{ii})$ donne exactement l'erreur *leave-one-out* sans refaire $n$ estimations, car $\hat m$ est linéaire en $y$. Le coût reste en $O(n^2)$.
- **Intervalle de confiance ponctuel** : intervalle $\hat m(x) \pm 1.96 \, \widehat{\text{se}}(x)$ valable séparément en chaque $x$. Avec $\ell_i(x)$ les poids de l'estimateur linéaire en $x$ et $e_i$ les résidus, l'écart-type robuste à l'hétéroscédasticité est
  $$\widehat{\text{se}}(x)^2 = \sum_i \ell_i(x)^2 \, e_i^2$$
  Il ne tient pas compte du biais de lissage. Une **bande uniforme** couvrirait toute la courbe à la fois, ce qui n'est pas fait ici.

### Modèle partiellement linéaire (Robinson)

- **Semiparamétrique** : une partie du modèle est paramétrique (des coefficients à estimer) et une partie est non paramétrique (une fonction inconnue).
- **Modèle partiellement linéaire** :
  $$y_i = x_i^\top\beta + g(z_i) + \varepsilon_i$$
  où $g$ est une fonction inconnue de $z$. Si $x$ dépend de $z$, les MCO qui oublient $g(z)$ souffrent d'un biais de variable omise (le 1.01 au lieu de 1.5 dans l'exemple).
- **Estimateur de Robinson (1988)** : en prenant l'espérance conditionnelle à $z$, $g(z)$ disparaît par différence,
  $$y_i - E[y \mid z_i] = \big(x_i - E[x \mid z_i]\big)^\top\beta + \varepsilon_i$$
  On estime donc $E[y \mid z]$ et $E[x \mid z]$ par régression à noyau, puis $\beta$ par MCO sur les résidus $\tilde y_i$ et $\tilde x_i$ :
  $$\hat\beta = \Big( \sum_i \tilde x_i \tilde x_i^\top \Big)^{-1} \sum_i \tilde x_i \tilde y_i$$
  Bien que $g$ soit estimée sans forme imposée, $\hat\beta$ converge à la **vitesse $\sqrt{n}$** (comme un estimateur paramétrique) et est asymptotiquement normal.
- **Troncature** : certains estimateurs écartent les observations où la densité de $z$ est très faible, pour stabiliser le calcul. Non faite ici.

### Vocabulaire de la complexité

- **$O(\cdot)$** : ordre de grandeur du nombre d'opérations (temps) ou de la place utilisée (mémoire), en ignorant les constantes. Un coût en $O(n^2)$ est multiplié par 4 quand $n$ double.
- **$n, k, G, d, B, q, n_h, I$** : nombre d'observations, de régresseurs, de points de la grille d'estimation, degré du polynôme local, nombre de tirages bootstrap, nombre de quantiles testés, nombre de fenêtres essayées et nombre d'itérations du solveur. Un **bloc** est un paquet de points traités ensemble pour limiter la mémoire.
