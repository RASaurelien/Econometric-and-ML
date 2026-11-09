## Définition

Notations : $i = 1, \dots, N$ indexe les unités (individus, entreprises), $t = 1, \dots, T$ les périodes, $y_{it}$ la variable expliquée, $x_{it}$ le régresseur, $\beta$ son coefficient (vrai $\beta = 1$) et $e_{it}$ le terme d'erreur. $\alpha_i$ est l'effet de l'unité $i$, $\gamma_t$ l'effet de la période $t$, $\bar x_i$ la moyenne de $x$ de l'unité $i$ sur le temps et $\bar x_t$ la moyenne de $x$ de la période $t$ sur les unités. Un chapeau ($\hat\beta$) désigne une valeur estimée.

### Cadre et problèmes à traiter

- **Données de panel** : on observe les mêmes unités sur plusieurs périodes. Chaque couple $(i, t)$ est une **cellule**.
- **Panel déséquilibré** : certaines cellules ne sont pas observées (ici chacune l'est avec probabilité 0.15 dans le scénario à peu d'observations).
- **Hétérogénéité non observée** : caractéristiques propres à une unité ou à une période, qu'on ne mesure pas mais qui influencent $y$.
- **Biais de corrélation avec les effets** : si $x$ est corrélé aux effets $\alpha_i$ (ou $\gamma_t$), un modèle qui les ignore attribue à $\beta$ une partie de leur influence. Les MCO empilés sont alors biaisés.
- **Modèle additif** : les effets d'unité et de période s'ajoutent.
  $$y_{it} = \beta x_{it} + \alpha_i + \gamma_t + e_{it}$$
- **Modèle interactif** : on ajoute des facteurs communs $f_t$ (de dimension $r$, le **rang**) dont l'effet varie selon l'unité par des poids $\lambda_i$. Ce terme n'est pas séparable en un effet d'unité plus un effet de période.
  $$y_{it} = \beta x_{it} + \alpha_i + \gamma_t + \lambda_i^\top f_t + e_{it}$$
- **Variable indicatrice** (*dummy*) : variable qui vaut 1 pour une unité (ou une période) et 0 sinon. Avoir un effet fixe par unité revient à ajouter une indicatrice par unité.
- **Variable catégorielle à forte cardinalité** : variable à très nombreuses modalités (ici les $N$ unités). Les indicatrices deviennent nombreuses, d'où l'idée de les remplacer par des embeddings.
- **Bruit irréductible** : part de $y$ qu'aucun modèle ne peut prévoir ($e_{it}$ de variance 1). La meilleure RMSE de prévision possible est donc 1.
- **Cellules d'entraînement, de validation et de test** : 20 % des cellules observées sont mises de côté pour mesurer la prévision hors échantillon. Un échantillon de validation sert à régler la pénalité et le nombre d'époques.

### Estimateurs

- **MCO empilés** (*pooled OLS*) : régression de $y$ sur $x$ en ignorant la structure de panel.
  $$\hat\beta = (X^\top X)^{-1} X^\top y$$
- **Effets fixes** (*fixed effects*) : $\alpha_i$ et $\gamma_t$ sont des paramètres libres, donc possiblement corrélés à $x$.
- **TWFE** (*two-way fixed effects*, effets fixes individuels et temporels) : effets fixes sur les deux dimensions. Pour un panel équilibré, on retire les moyennes par unité et par période, puis on régresse :
  $$\ddot y_{it} = y_{it} - \bar y_i - \bar y_t + \bar y, \qquad \hat\beta = \frac{\sum_{i,t} \ddot x_{it} \ddot y_{it}}{\sum_{i,t} \ddot x_{it}^2}$$
- **Projections alternées** : pour un panel déséquilibré, les moyennes ne suffisent plus. On retire à tour de rôle les moyennes par unité puis par période, en boucle, jusqu'à convergence. Le résultat est le même que celui de la régression avec indicatrices.
- **Effets fixes interactifs de Bai (2009)** : estimation du modèle interactif de rang $r$ en minimisant la somme des carrés des résidus sur $\beta$, $\lambda_i$ et $f_t$.
  $$\min_{\beta, \lambda, f} \; \sum_{i,t} \big( y_{it} - \beta x_{it} - \lambda_i^\top f_t \big)^2$$
- **Moindres carrés alternés** : on fixe $f$ pour estimer $\beta$ et $\lambda$ par MCO, puis on fixe $\lambda$ pour estimer $f$, et on répète jusqu'à convergence.
- **Choix du rang** : en pratique $r$ est inconnu et se choisit par critères d'information ou validation croisée. Ici il est fixé au vrai rang (2).
- **Effets aléatoires pénalisés (ridge)** : $\alpha_i$ et $\gamma_t$ sont estimés avec une pénalité $\ell_2$ qui les rapproche de zéro. Les effets deviennent traités comme des variables aléatoires de moyenne nulle, et la pénalité reflète le rapport entre la variance du bruit et celle des effets.
  $$\min_{\beta, \alpha, \gamma} \; \sum_{i,t} \big( y_{it} - \beta x_{it} - \alpha_i - \gamma_t \big)^2 + \lambda_\alpha \sum_i \alpha_i^2 + \lambda_\gamma \sum_t \gamma_t^2$$
- **Pénalité $\ell_2$** : terme qui ajoute au critère la somme des carrés des coefficients, ce qui les réduit. On la règle sur un échantillon de validation.
- **Variables de Mundlak (1978)** : on ajoute à la régression les moyennes de $x$ par unité et par période. Cela capte la corrélation entre $x$ et les effets, et élimine le biais des effets aléatoires. C'est le cas particulier de l'approche des effets aléatoires corrélés de Chamberlain (1982).
  $$y_{it} = \beta x_{it} + \delta_1 \bar x_i + \delta_2 \bar x_t + \alpha_i + \gamma_t + e_{it}$$
- **Test de Hausman** (1978) : compare l'estimateur à effets fixes (cohérent même si $x$ est corrélé aux effets) à celui à effets aléatoires (plus efficace s'il ne l'est pas). Un écart important indique que les effets sont corrélés à $x$.
  $$H = (\hat\beta_{FE} - \hat\beta_{RE})^\top \big[ \widehat{\text{Var}}(\hat\beta_{FE}) - \widehat{\text{Var}}(\hat\beta_{RE}) \big]^{-1} (\hat\beta_{FE} - \hat\beta_{RE})$$

### Embeddings

- **Embedding** : vecteur de nombres appris pour représenter chaque modalité d'une variable catégorielle (Guo et Berkhahn, 2016). Ici un vecteur $u_i$ par unité et un vecteur $v_t$ par période.
- **Embedding additif** : chaque vecteur est de dimension 1, donc ce sont exactement les effets $\alpha_i$ et $\gamma_t$. Avec une pénalité, le modèle est celui des effets aléatoires pénalisés, optimisé par descente de gradient au lieu d'une résolution exacte.
- **Embedding interactif** : les vecteurs $u_i$ et $v_t$ sont de dimension $d > 1$ et se combinent par produit scalaire, ce qui reproduit le terme interactif $\lambda_i^\top f_t$.
  $$\hat y_{it} = \hat\beta x_{it} + \hat\alpha_i + \hat\gamma_t + \langle u_i, v_t \rangle, \qquad \langle u_i, v_t \rangle = \sum_{k=1}^{d} u_{ik} \, v_{tk}$$
- **Produit scalaire** : somme des produits terme à terme de deux vecteurs, qui mesure leur alignement.
- **Descente de gradient** : on met à jour les paramètres $\theta$ en suivant l'opposé du gradient de la perte $\mathcal{L}$, avec un taux d'apprentissage $\eta$.
  $$\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}(\theta)$$
- **Adam** : variante de la descente de gradient qui adapte le pas de chaque paramètre à partir de moyennes mobiles du gradient $g_s$ ($m$) et de son carré ($v$), avec $\beta_1$ et $\beta_2$ proches de 1 et $\epsilon$ une petite constante.
  $$m_s = \beta_1 m_{s-1} + (1 - \beta_1) g_s, \quad v_s = \beta_2 v_{s-1} + (1 - \beta_2) g_s^2, \quad \theta \leftarrow \theta - \eta \frac{\hat m_s}{\sqrt{\hat v_s} + \epsilon}$$
  où $\hat m_s$ et $\hat v_s$ sont $m_s$ et $v_s$ corrigés de leur biais initial.
- **Entraînement en batch complet** : à chaque mise à jour, le gradient est calculé sur toutes les cellules d'entraînement, pas sur un sous-échantillon.
- **Époque** : un passage complet de l'entraînement sur les données. Le nombre d'époques se choisit sur l'échantillon de validation pour ne pas surajuster.
- **Surajustement** : le modèle colle trop aux données d'entraînement (bruit compris) et prévoit mal hors échantillon. Il apparaît quand le rang est trop élevé ou les données trop rares.
- **Régularisation** : toute contrainte (comme la pénalité $\ell_2$) qui limite la flexibilité du modèle pour réduire le surajustement. Elle aide quand il y a peu d'observations par unité.

### Évaluation

- **Répétitions** : on simule plusieurs panels à partir du même processus et on moyenne les résultats (15 ici). Chaque simulation utilise une graine fixée pour pouvoir la refaire.
- **Biais de $\beta$** : écart moyen entre l'estimation et la vraie valeur, sur $R$ répétitions.
  $$\text{biais} = \frac{1}{R} \sum_{r=1}^{R} \hat\beta_r - \beta$$
- **RMSE de $\beta$** (*root mean squared error*) : erreur totale, qui mélange biais et dispersion.
  $$\text{RMSE}_\beta = \sqrt{\frac{1}{R} \sum_{r=1}^{R} (\hat\beta_r - \beta)^2}$$
- **RMSE de prévision (test)** : erreur de prévision sur les cellules mises de côté, avec $n_{test}$ le nombre de ces cellules.
  $$\text{RMSE}_{test} = \sqrt{\frac{1}{n_{test}} \sum_{(i,t) \in \text{test}} (y_{it} - \hat y_{it})^2}$$
- **Écart-type clusterisé par unité** : écart-type robuste à une corrélation des erreurs à l'intérieur de chaque unité. Avec $X_i$ et $e_i$ les lignes de $X$ et les résidus de l'unité $i$ :
  $$(X^\top X)^{-1} \Big( \sum_{i=1}^{N} X_i^\top e_i e_i^\top X_i \Big) (X^\top X)^{-1}$$
- **Bootstrap par unité** : on tire des unités entières avec remise, on réestime, et on répète. L'écart-type des estimations donne l'écart-type de l'estimateur, utile pour les embeddings qui n'en ont pas.
- **Débiaisage** : correction du biais d'un estimateur régularisé pour pouvoir faire de l'inférence sur $\beta$.
- **Variable endogène retardée** (dynamique) : $y_{i,t-1}$ parmi les régresseurs. Elle est corrélée avec l'effet d'unité, et c'est exclu ici.
