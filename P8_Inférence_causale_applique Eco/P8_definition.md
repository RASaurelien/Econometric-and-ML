## Définition

Notations : $i$ indexe les unités (individus, comtés, régions) et $t$ les dates. $D_i$ vaut 1 si l'unité est traitée et 0 sinon. $Y_{it}(1)$ et $Y_{it}(0)$ sont les valeurs de la variable d'intérêt de l'unité $i$ à la date $t$ avec et sans traitement. $X_i$ désigne les covariables observées avant le traitement.

### Inférence causale

- **Inférence causale** : on cherche l'effet d'une politique ou d'un traitement, c'est-à-dire ce qui se serait passé *sans* traitement pour les unités traitées. Comme on n'observe jamais les deux situations pour la même unité, il faut un contrefactuel crédible.
- **Résultats potentiels** : chaque unité a deux valeurs possibles, $Y_i(1)$ si elle est traitée et $Y_i(0)$ sinon. On n'en observe qu'une. L'effet individuel $Y_i(1) - Y_i(0)$ n'est donc jamais observé.
- **ATT** (*average treatment effect on the treated*) : effet moyen du traitement sur les unités effectivement traitées,
  $$\text{ATT} = E\big[\, Y(1) - Y(0) \mid D = 1 \,\big]$$
- **Contrefactuel** : ce qu'aurait été $Y(0)$ pour les unités traitées. Chaque méthode ci-dessous propose une façon différente de le construire (des unités de comparaison, des unités juste de l'autre côté d'un seuil, une combinaison pondérée d'unités).
- **Effets hétérogènes** : l'effet du traitement varie selon les unités ou selon le temps écoulé depuis le traitement.

### Différences de différences

- **Différences de différences** (DiD) : on compare l'évolution du groupe traité avant et après le traitement à celle d'un groupe de contrôle. Dans le cas le plus simple (deux groupes, deux dates),
  $$\widehat{\text{ATT}} = \big(\bar Y_{traité, après} - \bar Y_{traité, avant}\big) - \big(\bar Y_{contrôle, après} - \bar Y_{contrôle, avant}\big)$$
  La première différence enlève ce qui est propre à chaque groupe et constant dans le temps, la seconde enlève ce qui est commun aux deux groupes dans le temps.
- **Hypothèse de tendances parallèles** : sans traitement, le groupe traité aurait évolué comme le groupe de contrôle. C'est l'hypothèse qui rend le DiD causal. Elle ne se teste pas directement, mais on regarde si les écarts *avant* le traitement sont proches de zéro.
- **Adoption échelonnée** : les unités sont traitées à des dates différentes.
- **Cohorte** (groupe $g$) : ensemble des unités traitées pour la première fois à la date $g$. Les unités **jamais traitées** forment un groupe à part. Les unités **pas encore traitées** à une date donnée peuvent aussi servir de contrôle.
- **ATT(g, t)** : effet du traitement à la date $t$ pour la cohorte $g$,
  $$\text{ATT}(g, t) = E\big[\, Y_t(1) - Y_t(0) \mid G = g \,\big]$$
  Estimé en comparant l'évolution de la cohorte $g$ entre une **période de base** et la date $t$ à celle des contrôles :
  $$\widehat{\text{ATT}}(g, t) = E[\, Y_t - Y_{base} \mid G = g \,] - E[\, Y_t - Y_{base} \mid \text{contrôle} \,]$$
- **Période de base** : la date de référence qui sert à calculer la variation. Avec l'option *universelle*, c'est toujours la période juste avant le traitement ($g - 1$), ce qui donne une cellule de référence à zéro. Avec l'option *variable*, c'est la période juste avant $t$ pour les dates antérieures au traitement et $g - 1$ pour les dates postérieures.
- **Trois estimateurs** (avec $\Delta Y = Y_t - Y_{base}$ et $D = 1$ pour la cohorte $g$, $0$ pour les contrôles) :
  - **Régression** (*outcome regression*) : on prédit l'évolution des traités sans traitement à partir des contrôles, $m(X) = E[\Delta Y \mid X, D = 0]$,
    $$\widehat{\text{ATT}}_{reg} = \frac{1}{n_1} \sum_{i : D_i = 1} \big( \Delta Y_i - \hat m(X_i) \big)$$
  - **IPW** (*inverse probability weighting*) : on repondère les contrôles pour qu'ils ressemblent aux traités. Avec le **score de propension** $p(X) = P(D = 1 \mid X)$, estimé par un logit,
    $$\widehat{\text{ATT}}_{IPW} = \frac{\sum_i D_i \, \Delta Y_i}{\sum_i D_i} - \frac{\sum_i \frac{\hat p(X_i)(1 - D_i)}{1 - \hat p(X_i)} \, \Delta Y_i}{\sum_i \frac{\hat p(X_i)(1 - D_i)}{1 - \hat p(X_i)}}$$
  - **Doublement robuste** (DR) : combine les deux, avec les poids $w_1 = D$ et $w_0 = \frac{\hat p (1 - D)}{1 - \hat p}$ appliqués à $\Delta Y - \hat m(X)$. Il est convergent si le modèle de score *ou* le modèle de régression est correct (pas besoin des deux).
- **Fonction d'influence** : fonction $\psi_i$ telle que $\hat\theta - \theta \approx \frac{1}{n} \sum_i \psi_i$. Elle permet de calculer l'écart-type et de combiner facilement plusieurs estimateurs (agrégations),
  $$\text{se}(\hat\theta) = \sqrt{\frac{1}{n} \cdot \frac{1}{n} \sum_{i=1}^n \hat\psi_i^2}$$
- **Agrégations** : les ATT(g, t) sont nombreux, on les résume de quatre façons, avec des poids proportionnels à la taille des cohortes $P(G = g)$ :
  - **Simple** : moyenne pondérée de tous les $\text{ATT}(g, t)$ après traitement ($t \ge g$).
  - **Dynamique** : on regroupe selon le temps relatif $e = t - g$ depuis le traitement, $\theta(e) = \sum_g w_g \, \text{ATT}(g, g + e)$. L'**effet dynamique** global est la moyenne des $\theta(e)$ pour $e \ge 0$.
  - **Par cohorte** : pour chaque $g$, $\theta(g) = \frac{1}{\#\{t \ge g\}} \sum_{t \ge g} \text{ATT}(g, t)$, puis moyenne pondérée entre cohortes.
  - **Par date** : pour chaque $t$, la moyenne des $\text{ATT}(g, t)$ des cohortes déjà traitées ($g \le t$), puis moyenne sur les dates.
- **Étude d'événement** (*event study*) : graphique des $\theta(e)$ en fonction du temps relatif $e$. Les points avant le traitement ($e < 0$) doivent être proches de zéro (tendances parallèles), les points après montrent l'évolution de l'effet.
- **DiD à effets fixes bi-directionnels** (TWFE) : régression classique avec un effet fixe par unité et par date,
  $$Y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \varepsilon_{it}$$
  Avec des effets hétérogènes et une adoption échelonnée, $\hat\beta$ est une moyenne pondérée de comparaisons 2×2 dont certaines utilisent des unités *déjà traitées* comme contrôles, avec des poids qui peuvent être négatifs (Goodman-Bacon, 2021). Il peut alors être fortement biaisé, voire de mauvais signe, ce que montre la simulation (0.89 au lieu de 2.47).

### Discontinuité de régression

- **Discontinuité de régression** (RDD) : le traitement dépend d'une **variable d'assignation** $X_i$ (variable de score) qui dépasse ou non un **seuil** $c$. Les unités juste au-dessus et juste en dessous sont comparables, donc le saut de $Y$ au seuil est un effet causal.
- **Discontinuité nette** (*sharp*) : le traitement est entièrement déterminé par le seuil, $D_i = \mathbb 1(X_i \ge c)$. L'effet estimé est
  $$\tau = \lim_{x \downarrow c} E[Y \mid X = x] - \lim_{x \uparrow c} E[Y \mid X = x]$$
- **Application du Sénat** (Lee, 2008) : $X$ est la marge de victoire du parti démocrate à l'élection précédente (seuil à 0) et $Y$ le résultat à l'élection suivante. $\tau$ mesure l'avantage d'avoir gagné de justesse.
- **Polynôme local** : on ajuste, de chaque côté du seuil, un polynôme de degré $p$ par moindres carrés pondérés, en ne gardant que les observations à moins de $h$ du seuil. Les poids sont ceux d'un **noyau triangulaire** $K(u) = (1 - |u|)\,\mathbb 1(|u| \le 1)$ avec $u = (x - c)/h$ : les observations proches du seuil pèsent plus. L'estimateur classique est $\hat\tau = \hat\mu_+ - \hat\mu_-$, la différence des deux ajustements en $c$.
- **Fenêtre** ($h$, *bandwidth*) : largeur de la zone autour du seuil. Petite, elle donne peu de biais et beaucoup de variance, grande l'inverse.
- **Biais de lissage et correction du biais** : le polynôme local de degré $p$ est biaisé car la vraie courbe n'est pas exactement un polynôme. On estime ce biais avec un polynôme de degré plus élevé $q = p + 1$, ajusté sur une autre fenêtre $b$, et on le soustrait,
  $$\hat\tau_{bc} = \hat\tau - \hat B$$
- **Variance robuste** : la variance de $\hat\tau_{bc}$ tient compte du fait que $\hat B$ est elle-même estimée (Calonico, Cattaneo et Titiunik, 2014). L'**intervalle robuste** à 95 % est $\hat\tau_{bc} \pm 1.96 \, \widehat{\text{se}}_{rb}$. Utiliser $\hat\tau_{bc}$ avec l'écart-type classique ferait sous-estimer l'incertitude.
- **Variance par plus proches voisins** : on estime la variance résiduelle de chaque observation par l'écart à la moyenne de ses $J = 3$ plus proches voisines du même côté du seuil,
  $$\hat\sigma_i^2 = \frac{J}{J + 1} \big( y_i - \bar y_{J(i)} \big)^2$$
- **Fenêtre optimale pour l'erreur quadratique moyenne** (MSE-optimale, type "mserd") : la fenêtre $h$ qui minimise l'erreur quadratique moyenne asymptotique de $\hat\tau$, qui arbitre entre le carré du biais et la variance,
  $$\text{AMSE}(h) = h^{2(p+1)} B^2 + \frac{V}{n h}$$
  Le minimum donne $h \propto n^{-1/(2p+3)}$. "rd" veut dire qu'une seule fenêtre est utilisée des deux côtés du seuil. $b$ est choisie de la même manière pour la correction du biais.
- **Points de masse** : valeurs de $X$ qui se répètent. Ils modifient le calcul des voisins et de la fenêtre, on utilise alors le nombre de valeurs distinctes de $X$.
- **Placebo (seuils fictifs)** : on refait l'estimation à de faux seuils, où il ne doit pas y avoir de saut, en n'utilisant que les observations d'un seul côté du vrai seuil. Un rejet fréquent signalerait une courbe irrégulière.
- **Test de manipulation** : si les unités peuvent choisir leur côté du seuil, la comparaison n'est plus valide. On regarde si les effectifs de part et d'autre du seuil, dans une fenêtre, sont équilibrés avec un test binomial de probabilité 1/2,
  $$n_{gauche} \sim \text{Binomiale}\big(n_{gauche} + n_{droite}, \tfrac{1}{2}\big) \quad \text{sous } H_0$$
  Le test habituel de la littérature (McCrary, 2008) compare les densités de $X$ de chaque côté.

### Contrôle synthétique

- **Contrôle synthétique** : une unité est traitée (le Pays basque) et on dispose de $J$ unités jamais traitées (les autres régions espagnoles). On construit une unité artificielle, moyenne pondérée des contrôles, qui imite la trajectoire de l'unité traitée *avant* le traitement. L'écart après traitement est l'effet estimé,
  $$\widehat{\text{effet}}_t = Y_{1t} - \sum_{j=2}^{J+1} w_j^* \, Y_{jt}, \qquad t \ge T_0$$
  où $T_0$ est la date du traitement.
- **Poids $W$** : un poids $w_j$ par unité de contrôle, positif, avec $\sum_j w_j = 1$. Ces contraintes évitent l'extrapolation, car le contrôle synthétique reste dans l'enveloppe convexe des contrôles.
- **Prédicteurs** : variables observées avant le traitement (éducation, investissement, part des secteurs, PIB moyen...) qu'on veut que l'unité synthétique reproduise. On note $X_1$ le vecteur de l'unité traitée et $X_0$ la matrice des contrôles. Un **prédicteur spécial** est la moyenne d'une variable sur une liste de périodes choisie. Les prédicteurs sont divisés par leur écart-type avant de les comparer.
- **Matrice $V$** : matrice diagonale qui donne un poids d'importance à chaque prédicteur. Pour un $V$ donné, les poids $W$ minimisent l'écart pondéré entre l'unité traitée et la combinaison de contrôles,
  $$W(V) = \arg\min_W \; (X_1 - X_0 W)^\top V \, (X_1 - X_0 W) \quad \text{sous} \quad w_j \ge 0, \; \sum_j w_j = 1$$
- **Programme quadratique** : ce problème est quadratique en $W$ avec des contraintes linéaires. Il s'écrit $\min_W \tfrac{1}{2} W^\top D \, W - d^\top W$ avec $D = X_0^\top V X_0$ et $d = X_0^\top V X_1$. Il est résolu par la fonction `solve.QP` du package `quadprog`. Quand il y a moins de prédicteurs que de contrôles, $D$ n'est pas inversible, d'où la petite constante ajoutée à sa diagonale.
- **Conditions de Kuhn-Tucker** : conditions qui caractérisent la solution d'un programme avec contraintes d'inégalité. Ici, le gradient $D W - d$ est égal à une même constante $\lambda$ pour tous les poids strictement positifs, et supérieur ou égal à $\lambda$ pour les poids nuls.
- **Choix de $V$** : $V$ est choisi pour que le contrôle synthétique suive au mieux la trajectoire de l'unité traitée avant le traitement, c'est-à-dire qu'il minimise l'erreur quadratique moyenne avant traitement sur les dates `time_optimize`,
  $$\min_V \; \frac{1}{|T_{opt}|} \sum_{t \in T_{opt}} \Big( Y_{1t} - \sum_j w_j(V) \, Y_{jt} \Big)^2$$
  Cette fonction n'est pas convexe, d'où un risque de **minimum local**.
- **Nelder-Mead** : algorithme d'optimisation qui n'utilise pas de dérivées (il fait évoluer un simplexe de points). On le relance depuis plusieurs **points de départ** (`s` dans la complexité) pour limiter le risque de minimum local.
- **Placebos dans l'espace** : on applique la même méthode à chaque unité de contrôle comme si elle avait été traitée (avec les autres contrôles comme donneurs). Si l'effet trouvé pour l'unité réellement traitée n'est pas plus extrême que ceux des placebos, il n'est pas significatif.
- **Rapport des RMSPE** : le RMSPE (*root mean squared prediction error*) est l'écart quadratique moyen entre l'unité et son contrôle synthétique. Pour chaque unité,
  $$\text{rapport} = \frac{\text{RMSPE}_{après}}{\text{RMSPE}_{avant}}, \qquad \text{p-value} = \frac{\#\{\text{unités avec un rapport} \ge \text{celui de l'unité traitée}\}}{\text{nombre total d'unités}}$$
  Avec 17 unités, la plus petite p-value possible est $1/17 \approx 0.06$. Le rapport corrige le fait que certaines unités sont mal ajustées avant le traitement. Ici, $8 / 17 \approx 0.47$ veut dire que 8 unités sur 17 ont un rapport au moins aussi grand.
- **Modèle à facteurs** : processus simulé $Y_{it} = \lambda_t^\top \mu_i + \varepsilon_{it}$, où chaque unité a des "chargements" $\mu_i$ sur des facteurs communs $\lambda_t$. Le contrôle synthétique fonctionne bien quand l'unité traitée est une combinaison convexe des contrôles, c'est le cas simulé dans les tests.

### Vocabulaire des tests et du package

- **Valeurs de référence** : résultats obtenus avec les packages R d'origine (`did`, `rdrobust`, `Synth`), auxquels on compare le code. `generer.R` les produit.
- **Tolérance** : écart relatif maximal accepté entre deux résultats pour dire qu'ils sont identiques (par exemple 1e-8). Elle tient compte des erreurs d'arrondi.
- **`testthat`, `DESCRIPTION`, `NAMESPACE`** : `testthat` est le package R qui exécute les tests. `DESCRIPTION` décrit le package (nom, version, dépendances) et `NAMESPACE` liste les fonctions qu'il exporte.
- **Couverture** : proportion des intervalles de confiance simulés qui contiennent la vraie valeur. Pour un intervalle à 95 %, elle devrait être proche de 95 %.

### Vocabulaire de la complexité

- **$O(\cdot)$** : ordre de grandeur du nombre d'opérations (temps) ou de la place utilisée (mémoire), en ignorant les constantes.
- **$n, T, G, k, I, c, m, s, J$** : nombre d'unités, de dates, de cohortes, de covariables, d'itérations du logit, d'unités de contrôle, de prédicteurs, de départs de l'optimiseur de $V$ et d'itérations de Nelder-Mead.
