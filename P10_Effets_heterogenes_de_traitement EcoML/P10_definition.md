## Définition

Notations : $Y$ est la variable de résultat (outcome), $T \in \{0, 1\}$ le traitement (1 si traité, 0 si contrôle), $X$ le vecteur des variables explicatives (covariables) et $n$ le nombre d'observations. $Y(1)$ et $Y(0)$ sont les résultats potentiels, c'est-à-dire ce qu'on observerait si l'individu était traité ou non. On n'en observe qu'un des deux, $Y = T \, Y(1) + (1 - T) \, Y(0)$. Un chapeau ($\hat\tau$) désigne une valeur estimée.

### Cadre et problèmes à traiter

- **ATE** (*average treatment effect*, effet moyen du traitement) : effet moyen sur toute la population.
  $$\text{ATE} = E[Y(1) - Y(0)]$$
- **CATE** (*conditional average treatment effect*, effet conditionnel moyen du traitement) : effet moyen pour les individus qui ont les caractéristiques $X = x$. C'est la quantité $\tau(x)$ estimée ici.
  $$\tau(x) = E[Y(1) - Y(0) \mid X = x]$$
- **Effet hétérogène** : l'effet du traitement n'est pas le même pour tout le monde, $\tau(x)$ varie avec $x$.
- **Ciblage** : donner le traitement en priorité aux individus dont $\hat\tau(x)$ est le plus élevé.
- **Traitement randomisé** : $T$ est tiré au hasard, indépendamment de $X$ (cas d'un A/B test). Le score de propension est alors constant.
- **Traitement observationnel, confusion** : $T$ n'est pas tiré au hasard et dépend de $X$, qui influence aussi $Y$. Comparer traités et contrôles mélange alors l'effet du traitement et l'effet de $X$.
- **Ignorabilité conditionnelle** : une fois $X$ fixé, le traitement est comme attribué au hasard. C'est l'hypothèse qui rend $\tau(x)$ identifiable en cas de confusion.
  $$(Y(0), Y(1)) \perp T \mid X$$
- **Score de propension** : probabilité d'être traité sachant $X$.
  $$e(x) = P(T = 1 \mid X = x)$$
- **Troncature du score** : on force $\hat e(x)$ à rester dans $[0.05, 0.95]$ pour éviter de diviser par un nombre proche de 0.
  $$\hat e(x) \leftarrow \min\big(\max(\hat e(x), 0.05), 0.95\big)$$

### Estimateurs

On note $\mu_t(x) = E[Y \mid X = x, T = t]$ la réponse moyenne du groupe $t$ et $\mu(x, t) = E[Y \mid X = x, T = t]$ la même fonction vue avec $t$ en argument.

- **Meta-learner** : méthode qui ramène l'estimation de $\tau(x)$ à des régressions classiques, avec n'importe quel modèle de ML au milieu.
- **S-learner** (*single*) : un seul modèle de $Y$ sur $(X, T)$, l'effet est la différence des prédictions avec $T = 1$ et $T = 0$.
  $$\hat\tau(x) = \hat\mu(x, 1) - \hat\mu(x, 0)$$
- **T-learner** (*two*) : un modèle par groupe, un sur les traités ($\hat\mu_1$) et un sur les contrôles ($\hat\mu_0$).
  $$\hat\tau(x) = \hat\mu_1(x) - \hat\mu_0(x)$$
- **X-learner** (Künzel et al., 2019) : on impute l'effet individuel de chaque observation avec le modèle de l'autre groupe, on régresse ces effets imputés sur $X$ dans chaque groupe, puis on combine les deux résultats avec le score de propension.
  $$\tilde D_i^1 = Y_i - \hat\mu_0(X_i) \;\; (T_i = 1), \qquad \tilde D_i^0 = \hat\mu_1(X_i) - Y_i \;\; (T_i = 0)$$
  $$\hat\tau(x) = \hat e(x) \, \hat\tau_0(x) + \big(1 - \hat e(x)\big) \, \hat\tau_1(x)$$
  où $\hat\tau_1$ et $\hat\tau_0$ sont les régressions de $\tilde D^1$ et de $\tilde D^0$ sur $X$.
- **DR-learner** (*doubly robust*) : on construit pour chaque individu un pseudo-outcome dont l'espérance conditionnelle vaut $\tau(x)$, puis on le régresse sur $X$.
  $$\varphi_i = \hat\mu_1(X_i) - \hat\mu_0(X_i) + \frac{T_i \big(Y_i - \hat\mu_1(X_i)\big)}{\hat e(X_i)} - \frac{(1 - T_i) \big(Y_i - \hat\mu_0(X_i)\big)}{1 - \hat e(X_i)}$$
- **Pseudo-outcome** : variable fabriquée à partir des données qui sert de cible dans une régression (ici $\varphi_i$).
- **Doublement robuste** : l'estimateur reste correct si le modèle de réponse ($\hat\mu_t$) ou le modèle de traitement ($\hat e$) est bien estimé, pas forcément les deux.
- **Cross-fitting** : on coupe l'échantillon en blocs, et les modèles utilisés pour construire $\varphi_i$ sont estimés sur les autres blocs que celui de l'observation $i$. Ça évite le surapprentissage.
- **Forêt causale** (Wager et Athey) : forêt d'arbres dont le but est d'estimer $\tau(x)$ et non de prédire $Y$. Chaque arbre regroupe des individus proches en $x$ (les feuilles) et mesure l'effet dans chaque groupe.
- **Outcome transformé** : variable qui, une fois moyennée dans une feuille, donne l'effet. Avec un score de propension connu $e$ :
  $$Y^* = Y \, \frac{T - e}{e (1 - e)}, \qquad E[Y^* \mid X = x] = \tau(x)$$
  Avec $e = 0.5$ : $Y^* = 2Y(2T - 1)$.
- **Honnêteté** : une moitié de l'échantillon sert à construire l'arbre (choisir les découpages), l'autre moitié à estimer l'effet dans les feuilles. Sans ça, l'effet estimé serait trop optimiste.
- **Effet dans une feuille** : différence de moyennes entre traités et contrôles de la feuille $L$, sur la moitié d'estimation.
  $$\hat\tau_L = \bar Y_{L, T = 1} - \bar Y_{L, T = 0}$$
- **Critère de découpage** : règle qui choisit où couper un arbre. Dans la version complète de Wager et Athey, il cherche à maximiser l'écart d'effet entre les deux nouvelles feuilles.
- **TARNet** (*treatment-agnostic representation network*) : réseau de neurones avec une représentation commune $\Phi(x)$ puis deux têtes, une par bras de traitement. Chaque observation n'entraîne que la tête de son propre groupe.
  $$\hat\tau(x) = h_1\big(\Phi(x)\big) - h_0\big(\Phi(x)\big)$$
- **Représentation, tête** : la représentation est un résumé appris de $X$, partagé par les deux groupes. Une tête est la dernière partie du réseau, qui prédit $Y$ à partir de ce résumé.

### Évaluation

- **PEHE** (*precision in estimation of heterogeneous effect*) : erreur de l'effet prédit par rapport au vrai effet, mesurable seulement en simulation car on connaît $\tau$. Ici c'est la racine de l'erreur quadratique moyenne.
  $$\text{PEHE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \big(\hat\tau(x_i) - \tau(x_i)\big)^2}$$
- **Corrélation de rang** (Spearman) : corrélation entre le classement des individus selon l'effet prédit et selon l'effet vrai. Elle vaut 1 si les classements sont identiques. Sans égalités, avec $d_i$ la différence de rang de l'individu $i$ :
  $$\rho = 1 - \frac{6 \sum_i d_i^2}{n (n^2 - 1)}$$
- **A/B test** : expérience où l'on tire $T$ au hasard et où l'on compare les résultats des deux groupes.
- **Échantillon test randomisé** : données où $T$ a été tiré au hasard, ce qui permet d'évaluer le ciblage sans connaître $\tau$.
- **Courbe de Qini** : on classe les individus par $\hat\tau$ décroissant, puis on regarde le gain cumulé quand on traite les $k$ premiers. Avec $S_k^T$ et $S_k^C$ les sommes de $Y$ chez les traités et chez les contrôles parmi les $k$ premiers, et $N_k^T$, $N_k^C$ leurs effectifs :
  $$Q(k) = S_k^T - S_k^C \, \frac{N_k^T}{N_k^C}$$
  Une courbe qui monte vite au début signifie que le modèle repère bien les individus sensibles au traitement.
- **AUUC** (*area under the uplift curve*) : aire sous la courbe de Qini. Plus elle est grande, meilleur est le ciblage.
- **Calibration par quintile** : on découpe les individus en 5 groupes selon $\hat\tau$, puis on compare dans chaque groupe l'effet moyen prédit à l'effet moyen observé (différence de moyennes traités et contrôles sur le test randomisé). Un modèle bien calibré a des points proches de la diagonale.
- **Intervalle de confiance** : plage de valeurs plausibles pour $\tau(x)$, de la forme $\hat\tau(x) \pm 1.96 \cdot \text{se}$ pour un niveau de 95 %.
- **Instable** : l'estimation change beaucoup pour une petite variation des données, par exemple quand $\hat e$ est proche de 0 ou de 1 et que les poids $1 / \hat e$ explosent.
