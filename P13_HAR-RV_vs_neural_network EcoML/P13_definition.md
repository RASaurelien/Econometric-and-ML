## Définition

Notations : $t$ indexe les jours, $j = 1, \dots, M$ les intervalles intraday d'une journée (par exemple $M = 78$ intervalles de 5 minutes), $P_{t,j}$ le prix et $h$ l'horizon de prévision en jours. $\text{RV}_t$ est la variance réalisée du jour $t$. $\hat f$ désigne une prévision et $\sigma^2_{res}$ la variance des résidus du HAR sur la fenêtre d'apprentissage.

### Données et mesures de volatilité

- **Données haute fréquence** : prix observés à l'intérieur de la journée (ici toutes les 5 minutes), au lieu d'un seul prix de clôture par jour.
- **Rééchantillonnage** : on garde un prix toutes les 5 minutes (`--freq`), ce qui réduit le bruit des prix très fréquents.
- **Rendement logarithmique intraday** : variation relative du prix d'un intervalle à l'autre. Les rendements sont calculés à l'intérieur de chaque journée.
  $$r_{t,j} = \ln P_{t,j} - \ln P_{t,j-1}$$
- **Volatilité** : ampleur des variations du prix. C'est l'écart-type des rendements, donc la racine de la variance.
- **Variance réalisée** (RV, *realized variance*) : mesure de la variance d'un jour obtenue en additionnant les carrés des rendements intraday (Andersen, Bollerslev, Diebold et Labys, 2003).
  $$\text{RV}_t = \sum_{j=1}^{M} r_{t,j}^2$$
- **Volatilité annualisée** : volatilité ramenée à l'année, avec environ 252 jours de bourse.
  $$\sigma_{ann} = \sqrt{252 \times \text{RV}_t}$$
- **Semi-variances** : on sépare la variance réalisée selon le signe du rendement. La semi-variance négative (*bad volatility*) ne garde que les baisses, la positive (*good volatility*) que les hausses.
  $$\text{RS}^-_t = \sum_{j} r_{t,j}^2 \, \mathbb{1}\{r_{t,j} < 0\}, \qquad \text{RS}^+_t = \sum_{j} r_{t,j}^2 \, \mathbb{1}\{r_{t,j} > 0\}, \qquad \text{RV}_t = \text{RS}^-_t + \text{RS}^+_t$$
- **Saut** : variation brutale et rare du prix, qui n'est pas due à la volatilité continue.
- **Variation bipuissance** (BV, *bipower variation*) : mesure de la variance qui ne réagit presque pas aux sauts, parce qu'elle multiplie les valeurs absolues de deux rendements consécutifs (Barndorff-Nielsen et Shephard, 2004).
  $$\text{BV}_t = \frac{\pi}{2} \sum_{j=2}^{M} |r_{t,j}| \, |r_{t,j-1}|$$
- **Composante de saut** : part de la variance qui vient des sauts. On la borne à zéro.
  $$J_t = \max(\text{RV}_t - \text{BV}_t, \; 0)$$
- **Noyaux réalisés, sous-échantillonnage** : estimateurs plus élaborés de la variance. Les noyaux pondèrent les autocovariances des rendements, le sous-échantillonnage fait la moyenne de RV calculées sur plusieurs grilles décalées. Le projet utilise les estimateurs simples ci-dessus.
- **Effet de levier** : une baisse du prix est suivie d'une hausse de la volatilité (corrélation négative entre rendement et volatilité future).
- **Saisonnalité intraday en U** : la volatilité est plus forte à l'ouverture et à la clôture qu'en milieu de journée.
- **Log-variance à deux composantes persistantes** : dans le simulateur, le logarithme de la variance est la somme d'un processus très persistant et d'un processus moins persistant.
- **Dérive** (*drift*) : tendance moyenne du prix, qui peut fausser la séparation entre hausses et baisses des semi-variances.

### Modèles

La cible est le logarithme de la variance réalisée moyenne sur les $h$ jours suivants.
$$y_{t+h} = \ln\!\Big( \frac{1}{h} \sum_{i=1}^{h} \text{RV}_{t+i} \Big)$$

- **Variables HAR** : variance réalisée du jour, de la semaine (5 jours) et du mois (22 jours).
  $$\text{RV}^{(d)}_t = \text{RV}_t, \qquad \text{RV}^{(w)}_t = \frac{1}{5} \sum_{i=0}^{4} \text{RV}_{t-i}, \qquad \text{RV}^{(m)}_t = \frac{1}{22} \sum_{i=0}^{21} \text{RV}_{t-i}$$
- **HAR-RV** (*heterogeneous autoregressive*, Corsi, 2009) : régression linéaire de la cible sur les trois variables HAR en logarithmes. Les trois horizons reproduisent la mémoire longue de la volatilité avec un modèle simple. C'est le benchmark de la littérature.
  $$y_{t+h} = \beta_0 + \beta_d \ln \text{RV}^{(d)}_t + \beta_w \ln \text{RV}^{(w)}_t + \beta_m \ln \text{RV}^{(m)}_t + \varepsilon_{t+h}$$
- **Mémoire longue** : les chocs de volatilité s'effacent très lentement, l'autocorrélation décroît bien plus lentement que dans un AR simple.
- **SHAR** (*semivariance HAR*, Patton et Sheppard, 2015) : le terme quotidien est remplacé par les deux semi-variances, ce qui laisse les baisses et les hausses avoir des effets différents.
  $$y_{t+h} = \beta_0 + \beta^- \ln \text{RS}^-_t + \beta^+ \ln \text{RS}^+_t + \beta_w \ln \text{RV}^{(w)}_t + \beta_m \ln \text{RV}^{(m)}_t + \varepsilon_{t+h}$$
- **Ridge sur 22 retards** : régression linéaire sur les 22 derniers $\ln \text{RV}$ quotidiens, avec pénalité $\ell_2$ qui réduit les coefficients, sans imposer la structure HAR. Le niveau de pénalité est $\lambda$.
  $$\hat\beta = \arg\min_\beta \; \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2 = (X^\top X + \lambda I)^{-1} X^\top y$$
- **Forêt aléatoire** : moyenne de $B$ arbres de décision construits sur des échantillons tirés au hasard.
  $$\hat f(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$
- **MLP** (*multilayer perceptron*) : réseau de neurones fait de couches linéaires séparées par des fonctions non linéaires, entraîné par descente de gradient.
- **LSTM** (*long short-term memory*) : réseau récurrent qui lit les 22 derniers jours pas à pas et garde une mémoire $c_t$ contrôlée par trois portes (oubli $f$, entrée $i$, sortie $o$). $\sigma$ est la sigmoïde et $\odot$ le produit terme à terme.
  $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c), \qquad h_t = o_t \odot \tanh(c_t)$$
- **Early stopping** : on arrête l'entraînement quand l'erreur sur un échantillon de validation (ici la fin de la fenêtre d'apprentissage) cesse de baisser, pour éviter le surapprentissage.
- **Graine** (*seed*) : valeur qui fixe l'aléa de l'initialisation du réseau. Moyenner trois graines réduit la variabilité due à l'entraînement.
- **Hyperparamètres** : réglages fixés avant l'entraînement (taille du réseau, nombre d'arbres, pénalité), ici choisis à la main.
- **Correction log-normale** : on prévoit le logarithme de la RV, mais on évalue aussi la RV en niveau. Si $\ln \text{RV}$ est normal de moyenne $f$ et de variance $\sigma^2_{res}$, la moyenne de $\text{RV}$ vaut :
  $$\widehat{\text{RV}} = \exp\!\Big( f + \frac{\sigma^2_{res}}{2} \Big)$$
  Sans cette correction, l'exponentielle de la prévision sous-estime la RV en niveau.

### Évaluation

- **Prévision hors échantillon** : prévision faite pour des dates qui n'ont pas servi à estimer le modèle.
- **Fenêtre glissante** : l'estimation se fait toujours sur les $W$ derniers jours (`--window`), et la fenêtre avance dans le temps. Elle diffère d'une fenêtre qui s'agrandit.
- **Réestimation** : fréquence à laquelle on refait l'estimation (chaque jour, tous les 5, 20 ou 40 jours selon le modèle), pour limiter le temps de calcul.
- **MSE** (*mean squared error*) : moyenne des erreurs au carré, sur le logarithme de la RV ou sur la RV en niveau.
  $$\text{MSE} = \frac{1}{P} \sum_{t} (y_t - \hat y_t)^2$$
- **Proxy bruité** : la vraie variance n'est pas observée, la RV n'en est qu'une mesure avec du bruit. Une perte est dite robuste si elle classe les modèles de la même façon avec ce proxy qu'avec la vraie variance.
- **QLIKE** : perte robuste au bruit de la mesure de volatilité (Patton, 2011), avec $v_t$ la RV observée et $\hat v_t$ la prévision en niveau.
  $$\text{QLIKE}_t = \frac{v_t}{\hat v_t} - \ln\!\Big( \frac{v_t}{\hat v_t} \Big) - 1$$
- **Test de Diebold-Mariano** : teste si deux modèles ont la même précision de prévision. Avec $d_t$ la différence de pertes (par exemple QLIKE du modèle moins QLIKE du HAR) et $\bar d$ sa moyenne :
  $$DM = \frac{\bar d}{\sqrt{\hat V(\bar d)}}, \qquad \hat V(\bar d) = \frac{1}{P} \Big( \hat\gamma_0 + 2 \sum_{k=1}^{h-1} \hat\gamma_k \Big)$$
  où $\hat\gamma_k$ est l'autocovariance de $d_t$ au retard $k$. Sous l'hypothèse nulle d'égalité, $DM$ suit approximativement une loi normale. La correction de Harvey, Leybourne et Newbold est un ajustement de petit échantillon qui multiplie $DM$ et compare à une loi de Student.
- **Cibles qui se chevauchent** : avec $h > 1$, deux dates consécutives partagent des jours dans leur cible, donc les erreurs sont autocorrélées. C'est pour cela que la variance du test utilise les autocovariances jusqu'à $h - 1$.
- **Comparaisons multiples** : quand on teste beaucoup de modèles contre le HAR, la probabilité d'obtenir au moins un résultat significatif par hasard augmente. Il faudrait une correction, qui n'est pas faite ici.
- **Régression de Mincer-Zarnowitz** : régression de la valeur observée sur la prévision. Une prévision sans biais a $a = 0$ et $b = 1$. Le $R^2$ mesure la part de variance expliquée.
  $$\text{RV}_t = a + b \, \hat v_t + u_t$$
- **Gain cumulé de QLIKE par rapport au HAR** : somme au fil des dates de la perte du HAR moins celle du modèle. La courbe monte quand le modèle bat le HAR.
  $$G_t = \sum_{s \leq t} \big( \text{QLIKE}_{HAR,s} - \text{QLIKE}_{mod,s} \big)$$
