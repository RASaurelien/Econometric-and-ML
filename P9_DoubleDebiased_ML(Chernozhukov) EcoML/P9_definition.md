## Définition

Notations : $Y$ est la variable de résultat, $D$ le traitement (ou la variable d'intérêt), $X$ le vecteur des $p$ variables de contrôle, $\theta$ le paramètre d'intérêt (l'effet causal de $D$ sur $Y$) et $\varepsilon$ le terme d'erreur. $n$ est le nombre d'observations et $K$ le nombre de blocs (folds) du cross-fitting. Un chapeau ($\hat\theta$) désigne une valeur estimée.

### Modèles et problèmes à traiter

- **Effet causal** : ce que devient $Y$ quand on change $D$ en gardant le reste identique. Une simple corrélation ne suffit pas, d'où le besoin de contrôler pour $X$.
- **Variables de contrôle, grande dimension** : les contrôles $X$ servent à comparer des individus comparables. On parle de grande dimension quand $p$ est grand par rapport à $n$ (ici $p = 300$ pour `plr_complex`), ce qui rend les MCO inutilisables.
- **Ignorabilité conditionnelle** : une fois qu'on contrôle pour $X$, le traitement est comme attribué au hasard. C'est l'hypothèse qui permet de lire $\theta$ comme un effet causal.
  $$(Y(0), Y(1)) \perp D \mid X$$
  où $Y(d)$ est le résultat potentiel si le traitement vaut $d$.
- **Fonction de nuisance** : fonction inconnue qu'il faut estimer pour arriver à $\theta$, mais qui n'est pas l'objet d'intérêt. Ici : $\ell(X) = E[Y \mid X]$, $m(X) = E[D \mid X]$ et, pour le modèle binaire, $g(d, X) = E[Y \mid D = d, X]$.
- **Modèle partiellement linéaire (PLR)** : l'effet de $D$ est linéaire, mais l'effet des contrôles est une fonction quelconque $g(X)$.
  $$Y = \theta D + g(X) + \varepsilon, \qquad D = m(X) + v, \qquad E[\varepsilon \mid D, X] = 0, \quad E[v \mid X] = 0$$
- **Modèle à traitement binaire (IRM)** (*interactive regression model*) : $D \in \{0, 1\}$ et l'effet peut dépendre de $X$.
  $$Y = g(D, X) + U, \qquad m(X) = P(D = 1 \mid X)$$
  La fonction $m(X)$ est le **score de propension** (probabilité d'être traité sachant $X$).
- **ATE** (*average treatment effect*, effet moyen du traitement) : effet moyen sur toute la population.
  $$\theta = E[Y(1) - Y(0)] = E[g(1, X) - g(0, X)]$$
- **Effet hétérogène** : l'effet du traitement varie d'un individu à l'autre selon $X$ (ici selon $X_1$). L'ATE en est la moyenne.
- **Corrélation de Toeplitz** : matrice de corrélation des régresseurs où la corrélation décroît avec la distance entre colonnes. Avec $\rho = 0.7$ :
  $$\Sigma_{jl} = \rho^{|j - l|}$$

### Estimateurs

- **DML** (*double/debiased machine learning*) : on estime les fonctions de nuisance avec du ML, puis on estime $\theta$ à partir d'un score insensible à leurs erreurs (score orthogonal). L'estimateur est asymptotiquement normal, donc l'inférence est valide :
  $$\sqrt{n}\,(\hat\theta - \theta) \xrightarrow{d} N(0, \sigma^2)$$
- **Partialling-out** : on retire de $Y$ et de $D$ la partie expliquée par $X$, puis on régresse les résidus l'un sur l'autre.
  $$\tilde Y_i = Y_i - \hat\ell(X_i), \qquad \tilde D_i = D_i - \hat m(X_i), \qquad \hat\theta = \frac{\sum_i \tilde D_i \tilde Y_i}{\sum_i \tilde D_i^2}$$
  Variance estimée et erreur standard :
  $$\hat\sigma^2 = \frac{\frac{1}{n}\sum_i \tilde D_i^2 (\tilde Y_i - \hat\theta \tilde D_i)^2}{\big(\frac{1}{n}\sum_i \tilde D_i^2\big)^2}, \qquad \text{se}(\hat\theta) = \sqrt{\hat\sigma^2 / n}$$
- **Cross-fitting** : on coupe l'échantillon en $K$ blocs. Les fonctions de nuisance appliquées au bloc $k$ sont estimées sur les $K - 1$ autres blocs, donc jamais sur les observations qu'elles servent à corriger. Ça évite le surapprentissage dans le score.
- **Cross-fitting stratifié sur le traitement** : même chose, mais chaque bloc garde la même part de traités et de non-traités que l'échantillon entier.
- **Cross-fitting répété** : on refait le découpage $S$ fois au hasard. On garde la médiane des estimations, avec une variance corrigée pour tenir compte de la variabilité due au découpage :
  $$\hat\theta = \operatorname{med}_s \hat\theta_s, \qquad \hat\sigma^2 = \operatorname{med}_s \big( \hat\sigma_s^2 + (\hat\theta_s - \hat\theta)^2 \big)$$
- **Score AIPW** (*augmented inverse probability weighting*) : score de l'ATE qui combine un modèle de résultat et un modèle de traitement. Il reste correct si l'un des deux est bien estimé (double robustesse).
  $$\psi_i = \hat g(1, X_i) - \hat g(0, X_i) + \frac{D_i \big(Y_i - \hat g(1, X_i)\big)}{\hat m(X_i)} - \frac{(1 - D_i) \big(Y_i - \hat g(0, X_i)\big)}{1 - \hat m(X_i)}$$
  $$\hat\theta = \frac{1}{n} \sum_i \psi_i, \qquad \hat\sigma^2 = \frac{1}{n} \sum_i (\psi_i - \hat\theta)^2$$
- **Estimateur agrégé** : l'équation du score est résolue une seule fois sur les $n$ observations (avec les prédictions issues du cross-fitting), au lieu de faire la moyenne de $K$ estimations par bloc.

### Learners de nuisance

Un *learner* est l'algorithme qui estime une fonction de nuisance.

- **Lasso (validation croisée)** : régression linéaire avec pénalité $\ell_1$, qui met à zéro les coefficients peu utiles. Le paramètre $\lambda$ est choisi par validation croisée (on garde le $\lambda$ qui prédit le mieux sur des blocs mis de côté).
  $$\hat\beta = \arg\min_\beta \; \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$
- **Random forest** (forêt aléatoire) : moyenne de $B$ arbres de décision construits sur des échantillons tirés au hasard.
  $$\hat f(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$
- **Gradient boosting** : on ajoute des petits arbres un par un, chacun corrigeant les erreurs du modèle précédent, avec un pas $\nu$.
  $$\hat f_t = \hat f_{t-1} + \nu \, h_t$$
- **Réseau de neurones (MLP)** : enchaînement de couches linéaires et de fonctions non linéaires, entraîné par descente de gradient.
- **Early stopping** : on arrête l'entraînement quand l'erreur sur l'échantillon de validation cesse de baisser, pour éviter le surapprentissage.
- **R2 hors échantillon** : part de variance de la cible expliquée sur des données non vues pendant l'entraînement.
  $$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$
- **Hyperparamètres** : réglages d'un learner fixés avant l'entraînement (nombre d'arbres, profondeur, taille du réseau). Le **réglage imbriqué** (*nested tuning*) les choisit par validation croisée à l'intérieur de chaque étape du cross-fitting.

### Méthodes de comparaison

- **OLS naïf** : régression de $Y$ sur $D$ seul. Biaisé dès que $X$ influence à la fois $D$ et $Y$.
- **OLS avec tous les contrôles** : régression de $Y$ sur $D$ et sur tout $X$. Correct si $g$ est linéaire, mais mauvais quand $p$ est grand ou $g$ non linéaire.
  $$\hat\beta = (W^\top W)^{-1} W^\top Y, \qquad W = [D, X]$$
- **Post-lasso** : un lasso sélectionne les contrôles utiles, puis on refait un OLS avec seulement les variables retenues.
- **Post-double-selection** (Belloni, Chernozhukov et Hansen) : un lasso de $Y$ sur $X$ et un lasso de $D$ sur $X$ sélectionnent chacun des contrôles, puis on fait un OLS de $Y$ sur $D$ et sur l'union des deux ensembles. Ça limite le risque d'oublier un contrôle qui influence surtout $D$.
- **Erreur standard robuste HC1** : version "sandwich" robuste à l'hétéroscédasticité, avec correction de degrés de liberté ($k$ est le nombre de régresseurs).
  $$\widehat{\text{Var}}(\hat\beta) = \frac{n}{n - k} (W^\top W)^{-1} \Big( \sum_i e_i^2 \, w_i w_i^\top \Big) (W^\top W)^{-1}$$

### Simulations et évaluation

- **Monte Carlo** : on simule $R$ jeux de données à partir d'un processus connu où la vraie valeur de $\theta$ est fixée, puis on regarde comment se comporte l'estimateur sur les $R$ répétitions.
- **Processus générateur de données** : programme qui fabrique les données simulées, avec les vraies fonctions de nuisance.
- **Biais** : écart moyen entre l'estimation et la vraie valeur.
  $$\text{biais} = \frac{1}{R} \sum_{r=1}^{R} \hat\theta_r - \theta$$
- **Écart-type** : dispersion des estimations d'une répétition à l'autre.
- **RMSE** (*root mean squared error*) : erreur totale, qui mélange biais et dispersion.
  $$\text{RMSE} = \sqrt{\frac{1}{R} \sum_{r=1}^{R} (\hat\theta_r - \theta)^2}$$
- **Erreur standard moyenne** : moyenne des $\text{se}(\hat\theta_r)$. Elle doit être proche de l'écart-type des estimations si l'inférence est bien calibrée.
- **Intervalle de confiance à 95 %** :
  $$\hat\theta \pm 1.96 \cdot \text{se}(\hat\theta)$$
- **Couverture** : part des répétitions où l'intervalle à 95 % contient la vraie valeur $\theta$. Elle doit être proche de 95 %. Avec $R$ répétitions, elle est bruitée (écart-type $\sqrt{0.95 \times 0.05 / R}$).
- **Estimateur sans biais avec les vraies nuisances** (oracle) : estimateur qui utilise les vraies fonctions $\ell$ et $m$ au lieu des estimées. Il sert de référence pour vérifier que le code est correct.
- **Bien calibré** : les erreurs standard et les intervalles annoncés correspondent à la vraie variabilité de l'estimateur (couverture proche de 95 %).

### Application

- **401(k)** : plan d'épargne retraite américain proposé par l'employeur. L'application étudie l'effet de l'**éligibilité** à ce plan sur le **patrimoine financier net** (actifs financiers moins dettes), avec les données de Poterba, Venti et Wise.
