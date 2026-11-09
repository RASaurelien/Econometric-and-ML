## Définition

Notations : le modèle simulé est $y_i = x_i^\top\beta + \varepsilon_i$ avec $n$ observations et $k$ régresseurs. Pour les variables instrumentales, $x$ est un seul régresseur endogène et $Z$ la matrice $n \times m$ des instruments. $R$ est le nombre de réplications (échantillons simulés) et $B$ le nombre de tirages bootstrap par réplication.

### Principe de la simulation

- **Monte Carlo** : on fabrique un grand nombre d'échantillons à partir d'un processus générateur de données dont on connaît les vrais paramètres, on applique l'estimateur ou le test à chacun, et on regarde la distribution des résultats. Cela permet de vérifier numériquement des propriétés que la théorie démontre seulement quand $n$ tend vers l'infini.
- **Réplication** : un échantillon simulé et l'estimation qui va avec. On en fait $R$.
- **Bruit de simulation** : une fréquence estimée sur $R$ réplications est elle-même aléatoire. Pour une probabilité $p$, son écart-type vaut
  $$\sqrt{p(1-p)/R}$$
  soit environ 0.5 point pour $p = 5\,\%$ et $R = 2000$. En dessous de cet écart, une différence n'est pas interprétable.
- **Vectorisation** : les estimateurs sont écrits pour traiter un tableau de $R$ échantillons à la fois, sans boucle Python. Les réplications sont traitées par **blocs** de taille $C$ pour ne pas dépasser la mémoire.
- **Processus générateur** : règle qui produit $(y, x, \varepsilon)$. Ici on peut y régler l'hétéroscédasticité, l'asymétrie des régresseurs et la forme des erreurs.

### Propriétés des estimateurs

- **Biais** : écart entre l'espérance de l'estimateur et la vraie valeur, $E[\hat\beta] - \beta$. On l'approche par la moyenne des $\hat\beta_r - \beta$ sur les réplications. Le **biais médian** utilise la médiane à la place de la moyenne (utile quand l'estimateur a des queues très épaisses, comme le 2SLS avec instruments faibles).
- **Consistance** : l'estimateur se rapproche de la vraie valeur quand $n$ grandit, $\hat\beta_n \xrightarrow{p} \beta$. Un estimateur biaisé peut être consistant (le biais disparaît avec $n$), et un estimateur peut ne pas être consistant (le biais reste).
- **Vitesse en racine de $n$** : l'écart-type d'un estimateur standard diminue comme $1/\sqrt{n}$, donc $\sqrt{n} \times \text{écart-type}$ se stabilise vers une constante quand $n$ grandit. C'est ce que le banc d'essai vérifie.
- **Normalité asymptotique** : la distribution de l'estimateur, une fois centrée et multipliée par $\sqrt{n}$, tend vers une loi normale même si les erreurs ne sont pas normales,
  $$\sqrt{n}(\hat\beta - \beta) \xrightarrow{d} N(0, V)$$
- **MCO endogène** : si $x$ est corrélé avec $\varepsilon$, les MCO ne convergent pas vers $\beta$ mais vers
  $$\text{plim}\,\hat\beta_{MCO} = \beta + \frac{\text{Cov}(x, \varepsilon)}{\text{Var}(x)}$$
  Dans l'exemple, $\text{Cov}(x,\varepsilon) = 0.6$ et $\text{Var}(x) = 1.48$, donc un biais asymptotique de $0.6 / 1.48 \approx 0.405$.
- **Statistique t robuste** : $t = (\hat\beta_k - \beta_k) / \text{se}_{HC}(\hat\beta_k)$, avec un écart-type robuste à l'hétéroscédasticité.

### Hétéroscédasticité et écarts-types

- **Hétéroscédasticité** : la variance des erreurs dépend des observations, $\text{Var}(\varepsilon_i \mid x_i) = \sigma_i^2$. L'écart-type classique $s^2 (X^\top X)^{-1}$ est alors faux.
- **Écart-type de White, HC0 à HC3** (*heteroskedasticity-consistent*) : sandwich de la forme
  $$(X^\top X)^{-1} \Big( \sum_i \omega_i \, e_i^2 \, x_i x_i^\top \Big) (X^\top X)^{-1}$$
  avec $e_i$ les résidus et $h_i = x_i^\top (X^\top X)^{-1} x_i$ le levier de l'observation $i$.

  | Version | $\omega_i$ |
  |---|---|
  | HC0 | $1$ |
  | HC1 | $n / (n - k)$ |
  | HC2 | $1 / (1 - h_i)$ |
  | HC3 | $1 / (1 - h_i)^2$ |

- **Levier** ($h_i$) : mesure à quel point $x_i$ est éloigné des autres observations. On a $\sum_i h_i = k$, donc un levier moyen de $k/n$. Un levier proche de 1 veut dire que l'observation tire fortement la droite vers elle, et que son résidu est trop petit.
- **Forts leviers** : situation où quelques observations ont un levier très supérieur à la moyenne, par exemple avec des régresseurs asymétriques. HC0 et HC1 sous-estiment alors la variance en petit échantillon, HC2 et HC3 corrigent ce défaut (MacKinnon et White, 1985).
- **Régresseur asymétrique** : dont la distribution a une longue queue d'un côté. Il produit quelques points extrêmes, donc des leviers très inégaux.

### Taille et puissance des tests

- **Niveau nominal** ($\alpha$, ici 5 %) : probabilité de rejet que le test est censé avoir sous l'hypothèse nulle.
- **Taille (empirique)** : proportion de réplications où le test rejette alors que $H_0$ est vraie,
  $$\text{taille} = \frac{1}{R} \sum_{r=1}^{R} \mathbb 1(\text{rejet dans la réplication } r)$$
  Un bon test a une taille proche du niveau nominal. Une taille de 50 % pour un niveau de 5 % veut dire que le test rejette à tort une fois sur deux. À l'inverse, un test qui rejette moins que $\alpha$ est dit **conservateur**.
- **Puissance** : proportion de rejets quand $H_0$ est fausse, pour un effet donné. Elle augmente avec la taille de l'effet et avec $n$. Elle n'est comparable entre tests que si leur taille est correcte (c'est la remarque faite sur le test classique dans les Limites).
- **Fonction de puissance** : la puissance tracée en fonction de la taille de l'effet.

### Bootstrap

- **Bootstrap** : méthode qui approche la distribution d'une statistique en refaisant l'estimation sur des échantillons tirés à partir des données elles-mêmes. Utile quand la loi asymptotique est mauvaise en petit échantillon.
- **Wild bootstrap** : on garde $X$ fixe et on recrée la variable expliquée en multipliant chaque résidu par un poids aléatoire $v_i$ d'espérance 0 et de variance 1 (par exemple $v_i = \pm 1$ avec probabilité 1/2, dite loi de Rademacher),
  $$y_i^* = x_i^\top\hat\beta + e_i \, v_i$$
  Comme le poids est propre à chaque observation, la forme de l'hétéroscédasticité est conservée. $X$ ne change pas, d'où un coût plus faible.
- **Bootstrap par paires** : on tire avec remise des couples $(y_i, x_i)$ entiers. Il ne suppose rien sur la forme de l'erreur, mais $X$ change à chaque tirage, d'où une régression à refaire à chaque fois.
- **Percentile-t** : on calcule dans chaque tirage une statistique t, $t_b^* = (\hat\beta_b^* - \hat\beta) / \text{se}_b^*$, puis on compare la vraie statistique t aux quantiles de ces $t_b^*$ au lieu de ceux d'une loi normale. On rejette si
  $$|t| > q_{1-\alpha}\big( |t_1^*|, \dots, |t_B^*| \big)$$
  Cette approche améliore la taille par rapport au bootstrap de l'estimateur seul.

### Instruments faibles

- **Variable instrumentale** : variable $z$ corrélée avec le régresseur endogène (pertinence) mais pas avec l'erreur (exogénéité).
- **Première étape** : régression du régresseur endogène sur les instruments, $x = Z\pi + v$.
- **2SLS** (doubles moindres carrés) : avec un seul régresseur endogène, $P_Z = Z(Z^\top Z)^{-1}Z^\top$,
  $$\hat\beta_{2SLS} = \frac{x^\top P_Z \, y}{x^\top P_Z \, x}$$
- **Instruments faibles** : les instruments ne sont presque pas corrélés avec $x$ ($\pi$ proche de 0). Le 2SLS est alors biaisé vers les MCO (le biais médian reste proche de celui des MCO), et ses tests t rejettent très souvent à tort, même avec $n$ grand.
- **Paramètre de concentration** ($\mu^2$, argument `mu2s`) : mesure globale de la force des instruments,
  $$\mu^2 = \frac{\pi^\top Z^\top Z \, \pi}{\sigma_v^2}$$
  Plus il est grand, plus les instruments sont forts.
- **F de première étape** : statistique de Fisher qui teste $\pi = 0$. Elle vaut en moyenne environ $1 + \mu^2 / m$ (par exemple 1.2 pour $\mu^2 = 1$ et $m = 5$). La règle usuelle est de se méfier si $F < 10$.
- **LIML** (*limited information maximum likelihood*) : estimateur de la famille des **estimateurs de classe $k$**,
  $$\hat\beta(k) = \frac{x^\top (I - k M_Z) \, y}{x^\top (I - k M_Z) \, x}, \qquad M_Z = I - P_Z$$
  Avec $k = 0$ on retrouve les MCO, avec $k = 1$ le 2SLS, et le LIML prend $k = \kappa$, la plus petite valeur propre de $(W^\top M_Z W)^{-1} W^\top W$ où $W = [\,y \;\; x\,]$. Il est beaucoup moins biaisé que le 2SLS quand les instruments sont faibles.
- **Test d'Anderson-Rubin** : teste $H_0 : \beta = \beta_0$ sans passer par l'estimation de $\beta$. On régresse $y - x\beta_0$ sur les instruments : si $H_0$ est vraie, les instruments n'expliquent rien. La statistique est
  $$AR(\beta_0) = \frac{(y - x\beta_0)^\top P_Z (y - x\beta_0) / m}{(y - x\beta_0)^\top M_Z (y - x\beta_0) / (n - m)} \sim F(m, \, n - m)$$
  sous des erreurs normales. Sa taille reste correcte quelle que soit la force des instruments, c'est son intérêt.

### Vocabulaire de la complexité

- **$O(\cdot)$** : ordre de grandeur du nombre d'opérations (temps) ou de la place utilisée (mémoire), en ignorant les constantes.
- **$R, n, k, B, m, C$** : nombre de réplications, d'observations, de régresseurs, de tirages bootstrap, d'instruments et taille d'un bloc de réplications traité à la fois.
