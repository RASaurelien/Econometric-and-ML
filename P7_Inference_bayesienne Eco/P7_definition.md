## Définition

Notations : $\theta$ désigne l'ensemble des paramètres du modèle et $y$ les données. $n$ est le nombre d'observations, $k$ le nombre de régresseurs, $N$ le nombre d'individus (dans le panel), $T_i$ le nombre d'observations de l'individu $i$ et $I$ le nombre d'itérations de l'échantillonneur. $\text{IG}(\alpha, \beta)$ est la loi inverse-gamma de densité proportionnelle à $x^{-\alpha - 1} e^{-\beta / x}$.

### Inférence bayésienne

- **Inférence bayésienne** : on traite les paramètres comme des variables aléatoires. On part d'une croyance initiale, qu'on met à jour avec les données grâce à la formule de Bayes, et on obtient une distribution complète sur les paramètres plutôt qu'une seule valeur estimée.
- **Prior** (loi *a priori*) : distribution $p(\theta)$ qui résume ce qu'on sait des paramètres avant de voir les données. Un prior **peu informatif** est très étalé (par exemple une normale d'écart-type 10) et laisse parler les données.
- **Vraisemblance** : $p(y \mid \theta)$, la probabilité des données pour une valeur donnée des paramètres.
- **Posterior** (loi *a posteriori*) : distribution des paramètres après avoir vu les données, par la formule de Bayes,
  $$p(\theta \mid y) = \frac{p(y \mid \theta) \, p(\theta)}{p(y)} \;\propto\; p(y \mid \theta) \, p(\theta)$$
  Le dénominateur $p(y)$ ne dépend pas de $\theta$ : on peut l'ignorer pour échantillonner.
- **Résumé a posteriori** : on décrit le posterior par sa moyenne, son écart-type, ses quantiles et un **intervalle de crédibilité** à 95 % (intervalle qui contient $\theta$ avec une probabilité de 95 % sachant les données, à ne pas confondre avec un intervalle de confiance).
- **Prior conjugué** : prior tel que le posterior est de la même famille. Par exemple, un prior normal sur les coefficients d'un modèle linéaire gaussien donne un posterior normal. C'est ce qui rend les lois conditionnelles du Gibbs exactes.
- **Loi inverse-gamma** : loi sur les réels positifs, utilisée pour les variances. Si $X \sim \text{Gamma}(\alpha, \text{taux } \beta)$ alors $1/X \sim \text{IG}(\alpha, \beta)$.

### MCMC (méthodes de Monte Carlo par chaînes de Markov)

- **Chaîne de Markov** : suite $\theta^{(1)}, \theta^{(2)}, \dots$ où chaque valeur ne dépend que de la précédente. Les méthodes MCMC construisent une chaîne dont la **loi stationnaire** est le posterior : une fois la chaîne assez longue, ses valeurs sont des tirages (corrélés) du posterior, et leur moyenne approche l'espérance a posteriori.
- **Burn-in** (`burn`) : nombre d'itérations du début de la chaîne qu'on jette, car la chaîne n'a pas encore atteint sa loi stationnaire.
- **Échantillonneur de Gibbs** : on tire chaque bloc de paramètres à tour de rôle, conditionnellement aux valeurs courantes des autres (sa **loi conditionnelle complète**). Toutes les conditionnelles doivent être faciles à tirer.
- **Gibbs pour le panel à effets aléatoires** : le modèle est
  $$y_{it} = x_{it}^\top b + a_i + e_{it}, \qquad a_i \sim N(0, \tau^2), \quad e_{it} \sim N(0, \sigma^2)$$
  avec les priors $b \sim N(b_0, B_0)$, $\sigma^2 \sim \text{IG}(s_1, s_2)$ et $\tau^2 \sim \text{IG}(t_1, t_2)$. Les quatre blocs ont des lois conditionnelles exactes :

  | Bloc | Loi conditionnelle |
  |---|---|
  | $a_i$ | $N(m_i, v_i)$ avec $v_i = \big(T_i/\sigma^2 + 1/\tau^2\big)^{-1}$ et $m_i = v_i \sum_t (y_{it} - x_{it}^\top b)/\sigma^2$ |
  | $b$ | $N(m, V)$ avec $V = \big(X^\top X/\sigma^2 + B_0^{-1}\big)^{-1}$ et $m = V\big(X^\top(y - a)/\sigma^2 + B_0^{-1} b_0\big)$ |
  | $\sigma^2$ | $\text{IG}\big(s_1 + n/2, \; s_2 + \tfrac{1}{2}\sum_{i,t}(y_{it} - x_{it}^\top b - a_i)^2\big)$ |
  | $\tau^2$ | $\text{IG}\big(t_1 + N/2, \; t_2 + \tfrac{1}{2}\sum_i a_i^2\big)$ |

  Ici $a$ désigne le vecteur des effets individuels répété sur chaque observation de l'individu. La matrice $X^\top X$ ne change pas d'une itération à l'autre, d'où son calcul unique.
- **Algorithme de Metropolis-Hastings** : quand les conditionnelles ne sont pas faciles à tirer (cas du logit), on **propose** une nouvelle valeur $\theta^*$ à partir de la valeur courante $\theta$, puis on l'accepte avec la probabilité
  $$\alpha = \min\Big( 1, \; \frac{p(\theta^* \mid y)}{p(\theta \mid y)} \Big)$$
  (formule valable pour une proposition symétrique) et sinon on garde $\theta$. Seul le rapport des posteriors compte, donc la constante $p(y)$ disparaît.
- **Marche aléatoire** : proposition de la forme $\theta^* = \theta + \varepsilon$ avec $\varepsilon \sim N(0, c^2 \Sigma)$ centré en 0, qui est bien symétrique.
- **Posterior du logit** : avec un prior normal $N(0, s^2 I)$ (où $s$ = `prior_sd`),
  $$\ln p(\theta \mid y) = \sum_i \big[ y_i \, x_i^\top\theta - \ln(1 + e^{x_i^\top\theta}) \big] - \frac{\lVert\theta\rVert^2}{2s^2} + \text{constante}$$
- **Approximation de Laplace** : on approche le posterior par une normale centrée sur son mode $\hat\theta$ (le **mode a posteriori**, trouvé par un algorithme de Newton) avec comme variance l'inverse de l'opposé du hessien en ce point,
  $$\Sigma_L = \Big( -\nabla^2 \ln p(\hat\theta \mid y) \Big)^{-1}$$
  Cette matrice donne la forme de la proposition du Metropolis-Hastings.
- **Règle $2.38/\sqrt{k}$** : l'échelle $c = 2.38/\sqrt{k}$ de la proposition est celle qui maximise l'efficacité pour une cible gaussienne en dimension $k$ (Roberts, Gelman et Gilks, 1997). Elle correspond à un **taux d'acceptation** (proportion de propositions acceptées) d'environ 0.44 en dimension 1 et 0.234 en grande dimension. Un taux trop bas veut dire des pas trop grands, un taux trop haut des pas trop petits.

### Diagnostics de convergence

- **Autocorrélation** : les tirages successifs d'une chaîne sont corrélés, donc ils apportent moins d'information que des tirages indépendants. On note $\rho_t$ la corrélation entre deux valeurs séparées de $t$ itérations.
- **Taille d'échantillon effective** (ESS) : nombre de tirages *indépendants* qui auraient la même précision que la chaîne,
  $$\text{ESS} = \frac{m I}{1 + 2 \sum_{t=1}^{\infty} \rho_t}$$
  pour $m$ chaînes de $I$ tirages. Pour un AR(1) de coefficient $\rho$, elle vaut approximativement $I \, (1 - \rho)/(1 + \rho)$ (c'est la valeur théorique utilisée dans le test).
- **Règle de Geyer** : la somme infinie des autocorrélations est tronquée à la première paire $\rho_{2j} + \rho_{2j+1}$ négative, car au-delà les estimations ne sont plus que du bruit.
- **R-hat** (de Gelman et Rubin) : compare la variance *entre* chaînes à la variance *dans* les chaînes, pour vérifier que plusieurs chaînes parties de points différents ont atteint la même distribution. Avec des chaînes de longueur $L$, $B$ la variance entre les moyennes des chaînes et $W$ la moyenne de leurs variances internes,
  $$\hat R = \sqrt{\frac{\frac{L-1}{L} W + \frac{1}{L} B}{W}}$$
  Une valeur proche de 1 (en dessous de 1.01 environ) indique que les chaînes sont cohérentes. Le **R-hat découpé** (*split*) coupe d'abord chaque chaîne en deux moitiés, ce qui permet aussi de détecter une chaîne qui dérive. La **normalisation par les rangs** (Vehtari et al., 2021) rend ce diagnostic robuste aux queues épaisses, elle n'est pas faite ici.
- **Test de Geweke** (de la distribution jointe) : test de correction du code. On alterne un tirage des paramètres sachant les données (un balayage de Gibbs) et un tirage de nouvelles données sachant les paramètres. Si le code est correct, la loi des paramètres ainsi obtenue est exactement le prior, et on compare ses moments simulés aux moments connus du prior.
- **Posterior par intégration numérique** : pour un logit à un seul paramètre, on calcule $p(\theta \mid y)$ sur une grille fine et on le normalise. On obtient le posterior exact, avec lequel on compare la moyenne, l'écart-type et les quantiles de la chaîne.
- **MixedLM, REML** : `MixedLM` est le modèle à effets mixtes de statsmodels. Il estime les variances par **REML** (maximum de vraisemblance restreint, qui corrige le biais de l'estimation des variances en tenant compte des degrés de liberté perdus pour estimer les coefficients). C'est la référence fréquentiste du panel.

### Générateur aléatoire

- **Graine** : nombre qui initialise le générateur. Une même graine redonne exactement la même suite de tirages, c'est ce qui permet de comparer les trois langages tirage par tirage.
- **xoshiro256\*\*** : générateur de nombres pseudo-aléatoires rapide, d'état de 256 bits (quatre mots de 64 bits) et de période $2^{256} - 1$ (Blackman et Vigna, 2018).
- **SplitMix64** : petit générateur utilisé uniquement pour transformer la graine de 64 bits en un état initial complet de xoshiro.
- **Box-Muller** : transforme deux uniformes indépendants $U_1, U_2 \in (0,1)$ en une normale centrée réduite,
  $$Z = \sqrt{-2 \ln U_1} \, \cos(2\pi U_2)$$
- **Marsaglia-Tsang** : méthode de **rejet** (on tire un candidat, on l'accepte ou le refuse selon un critère) pour tirer une loi gamma de forme $\alpha \ge 1$. Avec $d = \alpha - 1/3$ et $c = 1/\sqrt{9d}$, on tire $x \sim N(0,1)$, on pose $v = (1 + cx)^3$ et on accepte si $v > 0$ et $\ln U < x^2/2 + d - dv + d \ln v$, puis on renvoie $dv$. Pour une forme $\alpha < 1$, on tire une loi de forme $\alpha + 1$ qu'on multiplie par $U^{1/\alpha}$.
- **Tirage d'une inverse-gamma** : on tire $G \sim \text{Gamma}(\alpha, 1)$ et on renvoie $\beta / G$.

### Implémentations et compilation

- **Cœur** (*backend*) : la boucle d'échantillonnage elle-même, écrite en Rust, en C++ ou en Python. `backend="auto"` choisit le plus rapide disponible.
- **Compilation** : `g++` compile le C++ et `cargo` (le gestionnaire de projets de Rust) compile le Rust, sous forme de bibliothèques que Python appelle. `make` lance les deux.
- **libm** : bibliothèque mathématique standard du système (`exp`, `log`, ...). Les trois cœurs l'utilisent, d'où l'identité à 1e-15 près entre langages, à condition d'avoir la même.
- **Fonctions transcendantes, vectorisation** : les fonctions comme `exp` et `log1p` ne se calculent pas par de simples additions et multiplications. Numpy les applique à tout un tableau avec des instructions vectorielles (plusieurs valeurs par instruction), alors que les cœurs compilés les appellent une valeur à la fois, ce qui explique que le Metropolis-Hastings compilé soit plus lent pour n grand.

### Vocabulaire de la complexité

- **$O(\cdot)$** : ordre de grandeur du nombre d'opérations (temps) ou de la place utilisée (mémoire), en ignorant les constantes.
- **$n, k, N, I, m$** : nombre d'observations, de régresseurs, d'individus, d'itérations et de chaînes. `iter` dans la ligne « Mode et Laplace » est le nombre d'itérations de l'algorithme de Newton qui cherche le mode.
