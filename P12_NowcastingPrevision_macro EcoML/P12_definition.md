## Définition

Notations : $y_t$ est la variable à prévoir (mensuelle), $h$ l'horizon de prévision en mois, $X_t$ le vecteur des $N$ indicateurs du panel (environ 120) et $T$ le nombre de dates. $\hat y_{t+h|t}$ est la prévision de $y_{t+h}$ faite avec les données disponibles à la date $t$, et $e_{t+h} = y_{t+h} - \hat y_{t+h|t}$ l'erreur de prévision. $P$ est le nombre de prévisions évaluées.

### Cadre et données

- **Prévision macroéconomique** : prédire une variable macroéconomique future à partir de l'information disponible aujourd'hui.
- **Nowcasting** : prévoir le présent ou le très proche passé d'une variable dont la valeur officielle n'est publiée qu'avec retard (cas $h$ petit).
- **Pseudo temps réel** : à chaque date de test, les modèles sont estimés uniquement avec les observations antérieures, pour imiter un vrai prévisionniste.
- **Vintage, révision** : un vintage est la version d'une base de données telle qu'elle était publiée à une date donnée. Les statistiques officielles sont révisées après coup, donc un vintage récent contient des valeurs plus propres que celles qu'on avait en temps réel. Utiliser `current.csv` ignore donc les révisions, ce n'est pas du vrai temps réel.
- **FRED-MD** : base mensuelle de McCracken et Ng, environ 120 séries macroéconomiques américaines (production, emploi, prix, taux, monnaie).
- **INDPRO, CPIAUCSL** : codes FRED de la production industrielle et de l'indice des prix à la consommation.
- **Codes de transformation** : numéros qui disent comment rendre chaque série stationnaire (niveau, différence, log-différence, etc.).
  $$\Delta x_t = x_t - x_{t-1}, \qquad \Delta \log x_t = \log x_t - \log x_{t-1}$$
- **Écart interquartile** : écart entre le 3e et le 1er quartile, $\text{IQR} = Q_3 - Q_1$.
- **Bornage des valeurs extrêmes** : toute valeur à plus de 10 écarts interquartiles de la médiane est ramenée à cette limite.
  $$x \leftarrow \min\big(\max(x, \; \text{med} - 10\,\text{IQR}), \; \text{med} + 10\,\text{IQR}\big)$$
- **Remplissage par la dernière valeur connue** (*forward fill*) : une valeur manquante est remplacée par la dernière valeur observée de la même série.
- **Fuite d'information** (*look-ahead*) : utiliser, pour prévoir à la date $t$, une information qui n'était pas encore connue à $t$. Ici le seuil des valeurs extrêmes est calculé sur tout l'échantillon, d'où une petite fuite.
- **Panel à facteurs** : les nombreux indicateurs sont supposés dépendre d'un petit nombre de facteurs communs non observés $F_t$ (dimension $r$).
  $$X_t = \Lambda F_t + u_t$$
- **Facteurs principaux** (ACP, analyse en composantes principales) : estimation des facteurs. $\hat\Lambda$ est $\sqrt{N}$ fois les vecteurs propres de $X^\top X$ associés aux $r$ plus grandes valeurs propres, et les facteurs estimés valent
  $$\hat F = \frac{X \hat\Lambda}{N}$$
- **Réestimation** : fréquence à laquelle on réajuste le modèle pendant l'évaluation. Une réestimation moins fréquente (tous les 6 ou 12 mois) réduit le temps de calcul.

### Modèles

- **Benchmark** : modèle simple de référence auquel on compare les autres. Ici l'AR.
- **AR($p$)** (modèle autorégressif) : $y$ est expliqué par ses propres valeurs passées.
  $$y_t = c + \sum_{i=1}^{p} \phi_i \, y_{t-i} + \varepsilon_t$$
- **BIC** (*Bayesian information criterion*) : critère qui choisit l'ordre $p$ en équilibrant ajustement et nombre de paramètres $k$. On garde le $p$ qui minimise :
  $$\text{BIC} = T \ln\!\Big(\frac{\text{SCR}}{T}\Big) + k \ln T$$
  où SCR est la somme des carrés des résidus.
- **Indice de diffusion** (Stock et Watson, 2002) : prévision en deux étapes. On extrait les facteurs principaux du panel, puis on régresse la cible future sur ces facteurs.
  $$y_{t+h} = \alpha + \beta^\top \hat F_t + \varepsilon_{t+h}$$
- **Filtre de Kalman** : algorithme récursif qui estime une variable non observée (ici les facteurs) en alternant prédiction et mise à jour à chaque nouvelle donnée. Le projet ne l'utilise pas, la version à deux étapes est plus simple.
- **Elastic net** : régression pénalisée sur tout le panel, qui mélange pénalités $\ell_1$ et $\ell_2$ (`l1_ratio` est la part de $\ell_1$, $\rho_1$).
  $$\hat\beta = \arg\min_\beta \; \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \rho_1 \|\beta\|_1 + \frac{\alpha (1 - \rho_1)}{2} \|\beta\|_2^2$$
- **Validation croisée pour séries temporelles** : les blocs de validation viennent toujours après les blocs d'estimation, pour ne jamais prévoir le passé avec le futur.
- **LSTM** (*long short-term memory*) : réseau de neurones récurrent qui lit la série pas à pas et garde une mémoire $c_t$ contrôlée par trois portes ($f$ oubli, $i$ entrée, $o$ sortie), avec $\sigma$ la sigmoïde et $\odot$ le produit terme à terme.
  $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
  $$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c), \qquad h_t = o_t \odot \tanh(c_t)$$
- **Transformer** : réseau qui compare tous les mois de la fenêtre entre eux par un mécanisme d'attention, avec $Q$, $K$, $V$ des projections de l'entrée et $d_k$ leur dimension.
  $$\text{Attention}(Q, K, V) = \text{softmax}\!\Big(\frac{Q K^\top}{\sqrt{d_k}}\Big) V$$
- **Fenêtre de 12 mois** : le réseau reçoit les 12 derniers mois de facteurs principaux pour prévoir.
- **Graine** (*seed*) : valeur qui fixe l'aléa de l'initialisation du réseau. Moyenner les prévisions de plusieurs graines réduit la variabilité due à l'entraînement.

### Évaluation

- **Perte quadratique** : mesure l'erreur par son carré, $e_t^2$.
- **RMSE** (*root mean squared error*) :
  $$\text{RMSE} = \sqrt{\frac{1}{P} \sum_{t} e_t^2}$$
- **Test de Diebold-Mariano** : teste si deux modèles ont la même précision de prévision. Avec $d_t = e_{1,t}^2 - e_{2,t}^2$ la différence de pertes et $\bar d$ sa moyenne :
  $$DM = \frac{\bar d}{\sqrt{\hat V(\bar d)}}, \qquad \hat V(\bar d) = \frac{1}{P} \Big( \hat\gamma_0 + 2 \sum_{k=1}^{h-1} \hat\gamma_k \Big)$$
  où $\hat\gamma_k$ est l'autocovariance de $d_t$ au retard $k$. Sous l'hypothèse nulle d'égalité, $DM$ suit approximativement une loi normale. Une p-value faible indique que les deux modèles n'ont pas la même précision.
- **Correction de Harvey, Leybourne et Newbold** : correction de petit échantillon qui multiplie $DM$ et compare à une loi de Student à $P - 1$ degrés de liberté.
  $$DM^* = \sqrt{\frac{P + 1 - 2h + h(h-1)/P}{P}} \; DM$$
- **Gain cumulé d'erreur quadratique** : somme, au fil des dates, de l'erreur quadratique de l'AR moins celle du modèle. La courbe monte quand le modèle fait mieux que l'AR et descend quand il fait moins bien.
  $$G_t = \sum_{s \leq t} \big( e_{AR,s}^2 - e_{mod,s}^2 \big)$$
- **Récession** : période de baisse de l'activité économique, pendant laquelle les modèles riches en information gagnent souvent sur l'AR.
- **Model Confidence Set** : procédure qui garde l'ensemble des modèles dont la précision n'est pas significativement moins bonne que celle du meilleur, pour comparer tous les modèles à la fois (au lieu d'un test contre l'AR modèle par modèle).
