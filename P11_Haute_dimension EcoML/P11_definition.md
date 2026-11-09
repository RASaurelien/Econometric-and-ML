## Définition

Notations : $y$ est le vecteur des $n$ observations de la variable expliquée, $X$ la matrice $n \times p$ des régresseurs, $\beta$ le vecteur des coefficients, $\varepsilon$ le terme d'erreur de variance $\sigma^2$, $D$ la variable de traitement et $\theta$ son effet sur $y$. $\|\beta\|_1 = \sum_j |\beta_j|$ est la norme $\ell_1$ et $\|\beta\|_2^2 = \sum_j \beta_j^2$ la norme $\ell_2$ au carré. Un chapeau ($\hat\beta$) désigne une valeur estimée.

### Cadre et problèmes à traiter

- **Haute dimension** : le nombre de régresseurs $p$ est grand par rapport au nombre d'observations $n$ (ici $n = 500$, $p = 600$). Quand $p > n$, $X^\top X$ n'est pas inversible et les MCO n'existent plus.
- **Parcimonie** (*sparsity*) : seuls $s$ coefficients de $\beta$ sont non nuls (ici $s = 10$). Le vecteur $\beta$ est dit creux.
- **Support** : ensemble des indices des coefficients non nuls.
  $$S = \{ j : \beta_j \neq 0 \}$$
- **Parcimonie approximative** : les coefficients hors du support ne sont pas exactement nuls mais décroissent assez vite pour être négligeables. C'est ce que demande la théorie du post-double-selection, avec un taux de décroissance suffisant.
- **Corrélation de Toeplitz** : matrice de corrélation des régresseurs où la corrélation décroît avec la distance entre colonnes, avec $\rho \in \{0, 0.5, 0.9\}$.
  $$\Sigma_{jl} = \rho^{|j - l|}$$
- **Confusion** : $D$ dépend des mêmes variables que $y$. Comparer $y$ selon $D$ sans contrôler pour ces variables mélange l'effet de $D$ et celui des contrôles.
- **Modèle avec traitement** : $\theta$ est l'effet de $D$ quand $p$ contrôles sont disponibles.
  $$y = \theta D + X\beta + \varepsilon, \qquad D = X\gamma + v$$
- **Inférence valide** : l'intervalle de confiance annoncé contient bien la vraie valeur avec la probabilité annoncée, malgré la sélection de variables.

### Estimateurs

- **Lasso** (*least absolute shrinkage and selection operator*) : régression avec pénalité $\ell_1$. Elle réduit les coefficients et en met certains exactement à zéro, donc elle sélectionne des variables. Avec la paramétrisation de scikit-learn :
  $$\hat\beta_{lasso} = \arg\min_\beta \; \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \|\beta\|_1$$
  où $\alpha \geq 0$ est le niveau de pénalité.
- **Ridge** : régression avec pénalité $\ell_2$. Elle réduit tous les coefficients mais n'en annule aucun, donc elle ne sélectionne pas de variables.
  $$\hat\beta_{ridge} = \arg\min_\beta \; \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2 = (X^\top X + \lambda I)^{-1} X^\top y$$
- **Elastic net** : mélange des deux pénalités. `alpha` est le niveau de pénalité et `l1_ratio` (noté $\rho_1$) la part de $\ell_1$.
  $$\hat\beta_{EN} = \arg\min_\beta \; \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \rho_1 \|\beta\|_1 + \frac{\alpha (1 - \rho_1)}{2} \|\beta\|_2^2$$
- **Validation croisée** : on coupe l'échantillon en $K$ blocs, on estime sur $K - 1$ blocs, on mesure l'erreur de prévision sur le bloc restant, et on garde les paramètres de pénalité qui minimisent l'erreur moyenne. Elle est faite pour prévoir, pas pour retrouver le support.
- **Pénalité plug-in** (Belloni, Chen, Chernozhukov et Hansen, 2012) : niveau de pénalité calculé par une formule théorique, sans validation croisée, assez grand pour dominer le bruit. Dans la paramétrisation de scikit-learn, elle prend la forme
  $$\alpha = \frac{c \, \hat\sigma}{\sqrt{n}} \, \Phi^{-1}\!\Big(1 - \frac{\gamma}{2p}\Big)$$
  où $\Phi^{-1}$ est le quantile de la loi normale, $\gamma$ un petit niveau de confiance et $c > 1$ une constante. Elle est très parcimonieuse. Les constantes exactes du code peuvent différer, la forme est celle de la paramétrisation de scikit-learn.
- **Variance du bruit estimée de façon itérative** : $\sigma$ est inconnu. On part d'une première valeur, on calcule la pénalité, on estime le modèle, on recalcule $\hat\sigma$ avec les résidus, et on répète jusqu'à ce que $\hat\sigma$ se stabilise.
  $$\hat\sigma^2 = \frac{1}{n - \hat s} \sum_{i=1}^n e_i^2, \qquad e = y - X\hat\beta$$
  où $\hat s$ est le nombre de variables retenues.
- **OLS naïf** : régression de $y$ sur $D$ seul, sans contrôles. Biaisé à cause de la confusion.
- **Post-lasso** : un lasso sélectionne les variables, puis on refait un OLS avec seulement les variables retenues, ce qui enlève le biais de réduction du lasso.
- **Post-lasso naïf** : la sélection se fait sur l'équation de $y$ seulement. Une variable de contrôle qui influence surtout $D$ peut être oubliée, d'où un biais sur $\theta$.
- **Post-double-selection** (Belloni, Chernozhukov et Hansen, 2014) : un lasso de $y$ sur $X$ retient l'ensemble $\hat S_y$, un lasso de $D$ sur $X$ retient $\hat S_D$, puis on fait un OLS de $y$ sur $D$ et sur l'union des deux ensembles.
  $$\hat S = \hat S_y \cup \hat S_D, \qquad y \;\text{sur}\; (D, X_{\hat S})$$
  Un contrôle manqué dans l'équation de $y$ est souvent retrouvé dans celle de $D$, ce qui réduit le biais.
- **Erreur standard robuste HC1** : version "sandwich" robuste à l'hétéroscédasticité, avec une correction de degrés de liberté ($k$ est le nombre de régresseurs de la régression finale, $w_i$ une ligne du design).
  $$\widehat{\text{Var}}(\hat\beta) = \frac{n}{n - k} (W^\top W)^{-1} \Big( \sum_i e_i^2 \, w_i w_i^\top \Big) (W^\top W)^{-1}$$

### Mesures de sélection et de prévision

On note $\hat S$ le support estimé et $S$ le vrai support.

- **Précision** : part des variables retenues qui sont vraiment dans le support. Une précision faible veut dire beaucoup de variables inutiles retenues.
  $$\text{précision} = \frac{|\hat S \cap S|}{|\hat S|}$$
- **Rappel** : part des vraies variables qui ont été retrouvées. Un rappel inférieur à 1 veut dire qu'au moins une vraie variable a été manquée.
  $$\text{rappel} = \frac{|\hat S \cap S|}{|S|}$$
- **Erreur sur $\beta$** : distance entre les coefficients estimés et les vrais.
  $$\|\hat\beta - \beta\|_2 = \sqrt{\sum_j (\hat\beta_j - \beta_j)^2}$$
- **RMSE de prévision hors échantillon** (*root mean squared error*) : erreur de prévision sur $m$ observations qui n'ont pas servi à estimer le modèle.
  $$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat y_i)^2}$$
- **Coefficient faible** : coefficient non nul mais petit, difficile à distinguer du bruit. Le lasso avec pénalité plug-in peut le mettre à zéro.

### Simulations et inférence

- **Répétitions** (Monte Carlo) : on simule $R$ jeux de données à partir d'un processus connu où la vraie valeur $\theta$ est fixée, puis on regarde le comportement de l'estimateur sur les $R$ répétitions.
- **Biais** : écart moyen entre l'estimation et la vraie valeur.
  $$\text{biais} = \frac{1}{R} \sum_{r=1}^{R} \hat\theta_r - \theta$$
- **RMSE de l'effet de $D$** : erreur totale, qui mélange biais et dispersion.
  $$\text{RMSE} = \sqrt{\frac{1}{R} \sum_{r=1}^{R} (\hat\theta_r - \theta)^2}$$
- **Intervalle de confiance à 95 %** :
  $$\hat\theta \pm 1.96 \cdot \text{se}(\hat\theta)$$
- **Couverture** : part des répétitions où l'intervalle à 95 % contient la vraie valeur $\theta$. Elle doit être proche de 95 %. Avec $R$ répétitions elle est bruitée (écart-type $\sqrt{0.95 \times 0.05 / R}$).
- **Bien calibré** : les erreurs standard et les intervalles annoncés correspondent à la vraie variabilité de l'estimateur, donc la couverture est proche de 95 %.
- **Biais en échantillon fini** : biais qui disparaît quand $n$ grandit mais qui reste visible pour un $n$ donné, ici parce que les coefficients manqués par la sélection ne sont pas assez petits.
