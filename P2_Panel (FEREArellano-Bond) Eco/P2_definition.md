## Définition

Notations : $i = 1, \dots, N$ indexe les individus, $t = 1, \dots, T$ les périodes, $n$ est le nombre total d'observations ($n = NT$ si le panel est équilibré) et $K$ le nombre de régresseurs. Le modèle de base est

$$y_{it} = x_{it}^\top \beta + \alpha_i + \varepsilon_{it}$$

avec $\alpha_i$ l'effet individuel et $\varepsilon_{it}$ l'erreur idiosyncratique. Pour une variable $v$, $\bar v_i = \frac{1}{T_i} \sum_t v_{it}$ est la moyenne de l'individu $i$ et $\ddot v_{it} = v_{it} - \bar v_i$ la variable centrée (transformation within).

### Vocabulaire des données de panel

- **Panel** (ou données longitudinales) : on suit les mêmes $N$ individus sur plusieurs périodes. Cela permet de contrôler ce qui est propre à chaque individu et ne varie pas dans le temps.
- **Panel équilibré / déséquilibré** : équilibré si chaque individu est observé les $T$ mêmes périodes, déséquilibré sinon ($T_i$ varie).
- **Effet individuel** ($\alpha_i$) : caractéristique propre à l'individu, constante dans le temps et souvent non observée (talent, culture d'entreprise, géographie).
- **Exogénéité stricte** : l'erreur n'est corrélée avec les $x$ d'aucune période, $E[\varepsilon_{it} \mid x_{i1}, \dots, x_{iT}, \alpha_i] = 0$. C'est l'hypothèse faite sur les `X` dans la librairie.
- **Variable prédéterminée** : variable corrélée avec les erreurs passées mais pas avec les erreurs présentes et futures. Non gérée ici (voir Limites).
- **Hétérogénéité non observée** : le fait que les individus diffèrent par des $\alpha_i$ qu'on ne voit pas. Si $\alpha_i$ est corrélé avec $x_{it}$, les estimateurs qui l'ignorent sont biaisés.

### Modèles statiques

- **MCO empilés** (*pooled OLS*) : on empile toutes les observations et on fait un MCO comme s'il n'y avait pas d'effet individuel.
  $$\hat\beta_{MCO} = (X^\top X)^{-1} X^\top y$$
  Convergent seulement si $\alpha_i$ n'est pas corrélé avec les régresseurs.
- **Between** : MCO sur les $N$ moyennes individuelles, $\bar y_i = \bar x_i^\top \beta + \bar\alpha_i + \bar\varepsilon_i$. Utilise uniquement la variation entre individus.
- **Effets fixes**, **within** ou **FE** (*fixed effects*) : on centre chaque variable par sa moyenne individuelle, ce qui fait disparaître $\alpha_i$, puis on fait un MCO.
  $$\hat\beta_{FE} = (\ddot X^\top \ddot X)^{-1} \ddot X^\top \ddot y$$
  Utilise uniquement la variation dans le temps pour un même individu. Autorise la corrélation entre $\alpha_i$ et $x_{it}$, mais ne permet pas d'estimer l'effet des variables constantes dans le temps. Les degrés de liberté sont $n - N - K$.
- **Effets aléatoires**, **RE** (*random effects*) : on suppose $\alpha_i$ non corrélé avec $x_{it}$ et on fait un MCG qui combine variation within et between. En pratique, on fait un MCO sur les variables quasi-centrées
  $$y_{it} - \theta_i \bar y_i = (x_{it} - \theta_i \bar x_i)^\top \beta + \text{erreur}, \qquad \theta_i = 1 - \sqrt{\frac{\sigma_\varepsilon^2}{\sigma_\varepsilon^2 + T_i \sigma_\alpha^2}}$$
  Si $\theta_i = 0$ on retrouve les MCO empilés, si $\theta_i = 1$ on retrouve les effets fixes. Plus efficace que FE si l'hypothèse est vraie, biaisé sinon.
- **Swamy-Arora** : façon d'estimer les deux variances $\sigma_\varepsilon^2$ et $\sigma_\alpha^2$. On prend $\hat\sigma_\varepsilon^2$ dans les résidus des effets fixes, puis $\hat\sigma_\alpha^2 = \hat\sigma_B^2 - \hat\sigma_\varepsilon^2 / T$, où $\hat\sigma_B^2$ est la variance résiduelle de la régression between (avec une moyenne des $T_i$ à la place de $T$ si le panel est déséquilibré). C'est pour cela que les effets aléatoires demandent trois régressions.
- **Mundlak** : on ajoute les moyennes individuelles des régresseurs dans les MCO empilés,
  $$y_{it} = x_{it}^\top \beta + \bar x_i^\top \gamma + u_{it}$$
  Les pentes $\hat\beta$ sont égales à celles des effets fixes, et $\gamma = 0$ revient à dire que $\alpha_i$ n'est pas corrélé avec les $x$ (hypothèse des effets aléatoires).

### Modèle dynamique et Arellano-Bond

- **Modèle dynamique** : on ajoute la variable expliquée retardée,
  $$y_{it} = \rho \, y_{i,t-1} + x_{it}^\top \beta + \alpha_i + \varepsilon_{it}$$
  où $\rho$ mesure la persistance. `lags` est le nombre de retards de $y$.
- **Biais de Nickell** : dans ce modèle, $y_{i,t-1}$ est corrélé avec $\alpha_i$ donc, après centrage, avec l'erreur centrée. Les effets fixes sont alors biaisés vers le bas quand $T$ est petit, d'environ $-(1+\rho)/(T-1)$ pour $T$ grand, même avec $N$ très grand.
- **GMM en différences** (Arellano-Bond) : on passe aux différences premières, ce qui supprime $\alpha_i$,
  $$\Delta y_{it} = \rho \, \Delta y_{i,t-1} + \Delta x_{it}^\top \beta + \Delta\varepsilon_{it}$$
  mais $\Delta y_{i,t-1}$ est corrélé avec $\Delta\varepsilon_{it}$. On l'instrumente avec les niveaux plus anciens $y_{i,t-2}, y_{i,t-3}, \dots$, qui vérifient les conditions de moments
  $$E[\, y_{i,t-s} \, \Delta\varepsilon_{it} \,] = 0, \qquad s \geq 2$$
  Les $\Delta x_{it}$, supposés strictement exogènes, servent leurs propres instruments.
- **Instruments** : variables corrélées avec le régresseur endogène mais pas avec l'erreur. Sans `collapse`, il y a un instrument par couple (période, retard), soit environ $T^2/2$ colonnes.
- **`collapse`** : on ne garde qu'une colonne d'instruments par distance de retard $s$ (au lieu d'une par période et par retard), soit environ $T$ colonnes. Cela évite d'avoir trop d'instruments par rapport à $N$, ce qui affaiblit les tests.
- **GMM** (méthode des moments généralisés) : on choisit $\beta$ pour rendre les moments empiriques $\frac{1}{N}\sum_i Z_i^\top \Delta e_i$ proches de zéro, au sens d'une matrice de pondération $W$ :
  $$\hat\beta_{GMM} = (\Delta X^\top Z \, W \, Z^\top \Delta X)^{-1} \Delta X^\top Z \, W \, Z^\top \Delta y$$
  où $Z$ est la matrice d'instruments.
- **Une étape / deux étapes** : en une étape, $W$ est fixée à l'avance, $W_1 = \big(\sum_i Z_i^\top H Z_i\big)^{-1}$ avec $H$ la matrice qui a des 2 sur la diagonale et des $-1$ juste à côté (variance de $\Delta\varepsilon$ si les $\varepsilon$ sont indépendants). En deux étapes, on réutilise les résidus de la première étape pour obtenir le poids optimal
  $$W_2 = \Big( \sum_{i=1}^{N} Z_i^\top \Delta\hat e_i \, \Delta\hat e_i^\top Z_i \Big)^{-1}$$
- **Correction de Windmeijer** : en deux étapes, l'écart-type "naïf" est trop petit en échantillon fini, car il ignore que $W_2$ est elle-même estimée avec $\hat\beta$ de la première étape. Windmeijer (2005) ajoute un terme qui corrige ce défaut (la boucle sur les $K$ coefficients dans le tableau de complexité).
- **Anderson-Hsiao** : estimateur à variables instrumentales sur les différences premières, avec $y_{i,t-2}$ comme seul instrument. C'est le cas particulier d'Arellano-Bond avec $T = 3$ et sans $x$.
- **GMM en système** : variante (Blundell-Bond) qui ajoute les équations en niveaux. Non incluse.

### Écarts-types et p-values

- **Écart-type clusterisé** : robuste à une corrélation quelconque des erreurs à l'intérieur de chaque individu (ici, un cluster = un individu, $G = N$) et à l'hétéroscédasticité.
  $$\widehat{\text{Var}}(\hat\beta) = \frac{G}{G-1} \cdot \frac{n-1}{n-K} \cdot (X^\top X)^{-1} \Big( \sum_{g=1}^{G} X_g^\top e_g e_g^\top X_g \Big) (X^\top X)^{-1}$$
  Le facteur $\frac{G}{G-1}\frac{n-1}{n-K}$ est la correction de petit échantillon de Stata. `group_debias=True` dans `linearmodels` donne la même chose.
- **p-value** : probabilité, sous l'hypothèse nulle, d'obtenir une statistique au moins aussi extrême que celle observée. Pour les modèles statiques elle vient d'une loi de Student, pour Arellano-Bond d'une loi normale.

### Tests

- **Test de Hausman** : compare effets fixes (convergents dans tous les cas) et effets aléatoires (efficaces si $\alpha_i$ est non corrélé avec $x$). S'ils diffèrent, on rejette les effets aléatoires.
  $$H = (\hat\beta_{FE} - \hat\beta_{RE})^\top \big[ \widehat{\text{Var}}(\hat\beta_{FE}) - \widehat{\text{Var}}(\hat\beta_{RE}) \big]^{-1} (\hat\beta_{FE} - \hat\beta_{RE}) \sim \chi^2(K)$$
  Suppose des erreurs homoscédastiques.
- **Test de Mundlak robuste** : test de Wald de $H_0 : \gamma = 0$ dans la régression de Mundlak, avec une variance clusterisée. Il joue le même rôle que le Hausman mais reste valable avec de l'hétéroscédasticité ou de la corrélation intra-individu.
- **F de présence d'effets individuels** : teste $H_0 : \alpha_1 = \dots = \alpha_N$, c'est-à-dire que les MCO empilés suffisent. Avec $RSS$ la somme des carrés des résidus,
  $$F = \frac{(RSS_{MCO} - RSS_{FE}) / (N-1)}{RSS_{FE} / (n - N - K)} \sim F(N-1, \, n - N - K)$$
- **Test de Wald** : teste $q$ restrictions linéaires $R\beta = r$ à la fois.
  $$W = (R\hat\beta - r)^\top \big[ R \, \widehat{\text{Var}}(\hat\beta) \, R^\top \big]^{-1} (R\hat\beta - r) \sim \chi^2(q)$$
- **Tests AR(1) et AR(2)** d'Arellano-Bond : testent l'autocorrélation d'ordre 1 et 2 des résidus en différences $\Delta\hat e_{it}$. Une statistique $m_j$ suit approximativement une loi normale centrée réduite sous l'hypothèse nulle d'absence d'autocorrélation d'ordre $j$. On s'attend à rejeter AR(1), car $\Delta\varepsilon_{it}$ et $\Delta\varepsilon_{i,t-1}$ partagent $\varepsilon_{i,t-1}$ par construction. En revanche il ne faut pas rejeter AR(2), sinon les retards utilisés comme instruments ne sont plus valides.
- **J de Hansen** : test de sur-identification. Il vérifie que les moments en trop sont proches de zéro, c'est-à-dire que les instruments sont valides.
  $$J = \Big( \sum_i Z_i^\top \Delta\hat e_i \Big)^\top W_2 \Big( \sum_i Z_i^\top \Delta\hat e_i \Big) \sim \chi^2(m - p)$$
  où $m$ est le nombre d'instruments et $p$ le nombre de coefficients estimés. Une p-value faible suggère qu'au moins un instrument n'est pas valide. Un J qui vaut presque exactement 1 (p-value proche de 1) est suspect quand il y a trop d'instruments.

### Vocabulaire de la complexité

- **$O(\cdot)$** : ordre de grandeur du nombre d'opérations (temps) ou de la place utilisée (mémoire), en ignorant les constantes.
- **$n, N, T, K, m, q$** : nombre d'observations, d'individus, de périodes, de régresseurs, de colonnes de la matrice d'instruments et de restrictions testées par Wald.
