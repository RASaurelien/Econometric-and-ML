## Définition

Notations : $y_t$ est le vecteur des $k$ variables à la date $t$, $n$ le nombre de dates, $p$ le nombre de retards et $L$ l'opérateur retard ($L y_t = y_{t-1}$). $\Delta y_t = y_t - y_{t-1}$ est la différence première. $\Sigma$ désigne la matrice de variance-covariance des erreurs.

### Séries temporelles

- **Série temporelle** : suite d'observations ordonnées dans le temps. L'ordre compte, car les valeurs passées renseignent sur les valeurs présentes.
- **Stationnarité** : une série est (faiblement) stationnaire si sa moyenne, sa variance et ses autocovariances ne dépendent pas de $t$.
- **Racine unitaire**, **intégrée d'ordre 1**, $I(1)$ : une série non stationnaire (de type marche aléatoire, $y_t = y_{t-1} + u_t$) dont la différence première $\Delta y_t$ est stationnaire. Ses chocs ont des effets permanents.
- **Cointégration** : plusieurs séries $I(1)$ sont cointégrées si une combinaison linéaire d'entre elles est stationnaire, $\beta^\top y_t \sim I(0)$. Elles sont chacune non stationnaires mais restent liées par une relation d'équilibre de long terme (par exemple deux prix qui dérivent ensemble).
- **Retard** : valeur passée utilisée comme variable explicative. Un modèle à $p$ retards utilise $y_{t-1}, \dots, y_{t-p}$.

### VAR

- **VAR(p)** (vecteur autorégressif) : chaque variable dépend de ses propres retards et de ceux des autres variables,
  $$y_t = c + A_1 y_{t-1} + \dots + A_p y_{t-p} + u_t, \qquad u_t \sim (0, \Sigma)$$
  Estimé par MCO équation par équation. La matrice $\Sigma$ est estimée par $\hat\Sigma = \hat U^\top \hat U / (\text{degrés de liberté})$ à partir des résidus.
- **Stabilité** : le VAR est stable (donc stationnaire) si toutes les valeurs propres de sa matrice compagnon (qui empile $A_1, \dots, A_p$) sont de module strictement inférieur à 1.
- **Critères d'information** : servent à choisir $p$ en arbitrant entre qualité d'ajustement et nombre de paramètres. On prend le $p$ qui minimise le critère. Avec $\hat\Sigma_p$ la matrice estimée pour $p$ retards, $|\cdot|$ le déterminant et $n$ le nombre de dates utilisées :

  | Critère | Formule |
  |---|---|
  | AIC | $\ln\lvert\hat\Sigma_p\rvert + \dfrac{2}{n} \, p k^2$ |
  | BIC | $\ln\lvert\hat\Sigma_p\rvert + \dfrac{\ln n}{n} \, p k^2$ |
  | HQ | $\ln\lvert\hat\Sigma_p\rvert + \dfrac{2 \ln\ln n}{n} \, p k^2$ |
  | FPE | $\Big(\dfrac{n + kp + 1}{n - kp - 1}\Big)^k \lvert\hat\Sigma_p\rvert$ |

  Le BIC pénalise le plus, donc choisit les modèles les plus petits (c'est aussi le plus souvent convergent). L'AIC en choisit en général davantage. HQ est entre les deux. Les formules ci-dessus comptent $p k^2$ paramètres, `statsmodels` ajoute $k$ pour la constante, ce qui ne change pas le $p$ choisi car ce terme est le même pour tous les $p$.
- **Prévision** : on itère le VAR en remplaçant les valeurs futures inconnues par leurs prévisions, $\hat y_{t+1} = c + A_1 y_t + \dots + A_p y_{t-p+1}$, puis $\hat y_{t+2}$ avec $\hat y_{t+1}$, etc.
- **Causalité de Granger** : $x$ ne cause pas $y$ au sens de Granger si, une fois les retards de $y$ pris en compte, les retards de $x$ n'aident pas à prédire $y$. On teste $H_0$ : les coefficients des retards de $x$ dans l'équation de $y$ sont tous nuls (test de Wald ou de Fisher). Cela mesure un pouvoir prédictif, pas une causalité au sens strict.

### VAR structurel

- **Choc structurel** : choc économiquement interprétable ($\varepsilon_t$), non corrélé avec les autres et de variance 1. Les erreurs du VAR $u_t$ sont des mélanges de ces chocs, $u_t = P \varepsilon_t$ avec $\Sigma = P P^\top$. L'**identification** consiste à choisir $P$, car $\Sigma$ ne la détermine pas à elle seule.
- **Cholesky** : $P$ est la matrice triangulaire inférieure telle que $\Sigma = PP^\top$. Cela impose un ordre récursif : la première variable n'est pas affectée à la date même par les autres, la deuxième seulement par la première, etc. **Changer l'ordre des variables change les résultats.**
- **Identification de long terme** (Blanchard-Quah) : on impose des restrictions sur l'effet cumulé des chocs. Avec $C = (I - A_1 - \dots - A_p)^{-1}$, l'effet de long terme des chocs structurels est $C P$, et on choisit $P$ pour que cette matrice soit triangulaire inférieure, c'est-à-dire
  $$C P = \text{chol}(C \Sigma C^\top) \quad \Longleftrightarrow \quad P = C^{-1} \, \text{chol}(C \Sigma C^\top)$$
  Par exemple, un choc de demande n'a pas d'effet permanent sur la production.
- **Fonction de réponse impulsionnelle** (IRF) : effet, $h$ périodes plus tard, d'un choc structurel unitaire sur chaque variable. On calcule d'abord $\Psi_0 = I$ et $\Psi_h = \sum_{j=1}^{\min(h,p)} A_j \Psi_{h-j}$, puis
  $$\Theta_h = \Psi_h P$$
  L'élément $(i, j)$ de $\Theta_h$ est la réponse de la variable $i$ à un choc $j$ à l'horizon $h$.
- **Décomposition de variance** (FEVD) : part de la variance de l'erreur de prévision à $h$ pas de la variable $i$ qui vient du choc $j$,
  $$\omega_{ij,h} = \frac{\sum_{s=0}^{h-1} (\Theta_s)_{ij}^2}{\sum_{s=0}^{h-1} \sum_{l=1}^{k} (\Theta_s)_{il}^2}$$
- **Bootstrap des IRF** : on tire avec remise les résidus du VAR, on reconstruit une série en réappliquant le VAR, on réestime et on recalcule les IRF. En répétant $B$ fois, les quantiles (par exemple 5 % et 95 %) des $B$ IRF donnent un intervalle de confiance. C'est un **percentile simple** : sans correction du biais, il peut manquer sa couverture nominale.

### Cointégration de Johansen et VECM

- **VECM** (modèle à correction d'erreur) : réécriture d'un VAR de variables $I(1)$ en différences, avec un terme de retour vers l'équilibre,
  $$\Delta y_t = \Pi \, y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \, \Delta y_{t-i} + c + u_t, \qquad \Pi = \alpha\beta^\top$$
  - **$\beta$** ($k \times r$) : les $r$ **vecteurs de cointégration**, qui définissent les relations d'équilibre de long terme $\beta^\top y_{t-1}$.
  - **$\alpha$** ($k \times r$) : les **vitesses d'ajustement**, c'est-à-dire la réaction de chaque variable à l'écart d'équilibre de la période précédente.
  - **$\Gamma_i$** : dynamique de court terme.
  - **$r$** : le rang de $\Pi$, ou **rang de cointégration**. $r = 0$ : pas de cointégration (on travaille en différences), $r = k$ : toutes les séries sont stationnaires, $0 < r < k$ : cointégration.
- **Réécriture en VAR de niveaux** : un VECM correspond au VAR en niveaux avec $A_1 = I + \Pi + \Gamma_1$, $A_i = \Gamma_i - \Gamma_{i-1}$ pour $2 \le i \le p-1$ et $A_p = -\Gamma_{p-1}$. Cela permet de réutiliser le code des IRF.
- **Test de Johansen** : estime les valeurs propres $\hat\lambda_1 \ge \dots \ge \hat\lambda_k$ (corrélations canoniques au carré) et teste le rang. $H_0$ : le rang est au plus $r$.
  - **Statistique de la trace** : $\;\lambda_{trace}(r) = -n \sum_{i=r+1}^{k} \ln(1 - \hat\lambda_i)$
  - **Statistique de la valeur propre maximale** : $\;\lambda_{max}(r) = -n \ln(1 - \hat\lambda_{r+1})$
  - **Choix du rang** : on teste $r = 0$, puis $r = 1$, etc., et on retient le premier $r$ non rejeté.
- **Valeurs critiques simulées** : sous $H_0$, ces statistiques ne suivent pas une loi usuelle (χ²) mais une loi qui dépend de fonctionnelles de mouvements browniens. Les valeurs critiques sont donc obtenues par simulation de marches aléatoires, pour chaque dimension $d$ du système, `reps` fois.
- **Terme déterministe** : constante ou tendance. Ici, soit aucun, soit une **constante libre** (les niveaux peuvent avoir une dérive), sans constante dans la relation de cointégration elle-même.

### Filtre de Kalman

- **Modèle d'espace d'états** : on décrit une série observée $y_t$ à partir d'un **état** non observé $a_t$ (vecteur de $m$ composantes) qui évolue dans le temps,
  $$y_t = Z a_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, H) \qquad\qquad a_{t+1} = F a_t + \eta_t, \quad \eta_t \sim N(0, Q)$$
  $y_t$ a $q$ composantes observées. $Z$ est la matrice d'observation, $F$ la matrice de transition, $H$ et $Q$ les variances des bruits d'observation et d'état.
- **Filtre de Kalman** : algorithme récursif qui donne la meilleure estimation de l'état à la date $t$ avec l'information disponible jusque-là. Notons $a_{t|t-1}$ la prévision de l'état et $P_{t|t-1}$ sa variance. À chaque date :
  - **innovation** : $v_t = y_t - Z a_{t|t-1}$, de variance $S_t = Z P_{t|t-1} Z^\top + H$
  - **gain de Kalman** : $K_t = P_{t|t-1} Z^\top S_t^{-1}$
  - **mise à jour** : $a_{t|t} = a_{t|t-1} + K_t v_t$ et $P_{t|t} = P_{t|t-1} - K_t Z P_{t|t-1}$
  - **prédiction** : $a_{t+1|t} = F a_{t|t}$ et $P_{t+1|t} = F P_{t|t} F^\top + Q$
- **Log-vraisemblance** : construite à partir des innovations, ce qui permet l'estimation par **maximum de vraisemblance** des paramètres du modèle ($H$, $Q$, coefficients de l'ARMA...),
  $$\ell = -\frac{1}{2} \sum_{t} \Big[ q \ln 2\pi + \ln\lvert S_t \rvert + v_t^\top S_t^{-1} v_t \Big]$$
- **Valeurs manquantes** : si $y_t$ n'est pas observé, on saute l'étape de mise à jour ($a_{t|t} = a_{t|t-1}$, $P_{t|t} = P_{t|t-1}$) et la date ne contribue pas à la vraisemblance. L'incertitude sur l'état grossit alors jusqu'à la prochaine observation.
- **Lisseur de Rauch-Tung-Striebel** : repasse en arrière après le filtre pour estimer l'état en utilisant **toutes** les observations, passées et futures, d'où une estimation plus précise que le filtre seul,
  $$a_{t|n} = a_{t|t} + L_t \, (a_{t+1|n} - a_{t+1|t}), \qquad L_t = P_{t|t} F^\top P_{t+1|t}^{-1}$$
- **Initialisation diffuse** : quand on ne sait rien de l'état initial, on lui donne une très grande variance $P_{0} = 10^7 \, I$ (approximation de la version exacte).
- **Modèle de niveau local** : le plus simple des modèles d'états, une marche aléatoire observée avec du bruit,
  $$y_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma_\varepsilon^2) \qquad \mu_{t+1} = \mu_t + \eta_t, \quad \eta_t \sim N(0, \sigma_\eta^2)$$
  Le rapport $\sigma_\eta^2 / \sigma_\varepsilon^2$ règle la réactivité du niveau estimé : plus il est grand, plus le niveau suit de près les observations.
- **ARMA(1,1)** : $y_t = \phi \, y_{t-1} + \varepsilon_t + \theta \, \varepsilon_{t-1}$, qui s'écrit sous forme d'espace d'états et s'estime donc avec le filtre. La stationnarité demande $|\phi| < 1$, imposée ici par une pénalité dans l'optimisation.
- **EQM** (erreur quadratique moyenne) : moyenne des carrés des écarts entre l'estimation et la vraie valeur, utilisée pour comparer le niveau lissé aux observations brutes.

### Compilation (C++)

- **C++ et bibliothèque partagée** : le filtre est écrit en C++ pour aller plus vite, puis compilé en bibliothèque (`.so` sous Linux, `.dll` sous Windows) que Python appelle. Sans compilation, une version numpy plus lente prend le relais.
- **`g++`, `PATH`, Makefile** : `g++` est le compilateur, le `PATH` est la liste de dossiers où le système cherche les programmes (et ici les DLL d'exécution), et le Makefile automatise la commande de compilation.

### Vocabulaire de la complexité

- **$O(\cdot)$** : ordre de grandeur du nombre d'opérations (temps) ou de la place utilisée (mémoire), en ignorant les constantes.
- **$n, k, p, H, B, m, q, I$** : nombre de dates, de variables, de retards, horizon des IRF, nombre de tirages bootstrap, nombre d'états, nombre de variables observées et nombre d'itérations de l'optimiseur. Dans le tableau, $T$ est le nombre de dates de la série simulée, $d$ la dimension du système pour les valeurs critiques et `reps` le nombre de simulations.
