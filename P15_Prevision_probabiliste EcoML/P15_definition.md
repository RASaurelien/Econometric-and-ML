## Définition

Notations : $y_t$ est la valeur observée à la date $t$, $\hat y_t$ sa prévision ponctuelle et $T$ le nombre de dates de test. $\alpha$ est le niveau d'erreur visé (ici $\alpha = 0.10$, donc une couverture visée de $1 - \alpha = 0.90$). $C_t = [l_t, u_t]$ est l'intervalle de prédiction à la date $t$, de bornes $l_t$ et $u_t$. $s_i$ est le score de la date $i$ et $n$ le nombre de scores de calibration.

### Cadre et problèmes à traiter

- **Intervalle de prédiction** : intervalle dans lequel la vraie valeur future doit tomber avec une probabilité donnée. Il diffère d'un intervalle de confiance, qui porte sur un paramètre.
- **Couverture** : fréquence à laquelle l'intervalle contient la vraie valeur. Un intervalle à 90 % est bien calibré si sa couverture est proche de 0.90.
  $$\text{couverture} = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{ y_t \in C_t \}$$
- **Couverture empirique, niveau nominal** : la couverture empirique est celle mesurée sur les données, le niveau nominal est celui annoncé ($1 - \alpha$).
- **Prévision conforme** (*conformal prediction*, Vovk et al., 2005) : méthode qui construit des intervalles à partir des erreurs passées d'un modèle quelconque. Elle donne une garantie de couverture sans hypothèse sur la loi des erreurs (*distribution-free*).
- **Échangeabilité** : l'ordre des données n'a pas d'importance, c'est-à-dire que leur loi jointe ne change pas quand on les permute. C'est l'hypothèse de la garantie conforme. Elle est fausse pour une série temporelle qui change de régime.
- **Rupture, changement de régime** : à une date donnée, la loi de la série change (ici la volatilité double à 70 % de l'échantillon). Les erreurs passées ne ressemblent plus aux erreurs futures.
- **Régime calme, régime agité** : période de faible ou de forte volatilité.
- **Volatilité à deux régimes** : la variance de l'erreur prend deux niveaux selon le régime.
- **Erreurs de Student** : erreurs tirées d'une loi de Student, dont les queues sont plus épaisses que celles de la loi normale (plus de valeurs extrêmes).
- **AR non linéaire** : modèle autorégressif où la valeur courante est une fonction non linéaire des valeurs passées.
- **Garantie marginale, garantie conditionnelle** : une garantie marginale porte sur la couverture moyenne sur tout l'échantillon. Une garantie conditionnelle porte sur la couverture dans chaque sous-groupe (par exemple chaque régime). La conforme et ACI ne donnent que la première.

### Méthodes

- **Prévision ponctuelle** : valeur unique prédite pour la date suivante, ici par un AR ou du gradient boosting.
- **Échantillons d'estimation, de calibration et de test** : le modèle est estimé sur les 35 % premiers points, les scores sont calculés sur les 15 % suivants (calibration), et les intervalles sont évalués sur le reste (test), dans l'ordre temporel.
- **Intervalle gaussien** : référence naïve qui suppose des erreurs normales. Avec $\hat\sigma$ l'écart-type des résidus de calibration et 1.645 le quantile à 95 % de la loi normale :
  $$C_t = \big[ \hat y_t - 1.645 \, \hat\sigma, \; \hat y_t + 1.645 \, \hat\sigma \big]$$
- **Score de non-conformité** : mesure de l'écart entre la prévision et la réalité. Ici le résidu absolu.
  $$s_i = | y_i - \hat y_i |$$
- **Quantile conforme avec correction de taille finie** : on prend le quantile de niveau $\lceil (n + 1)(1 - \alpha) \rceil / n$ des scores de calibration, c'est-à-dire le $\lceil (n + 1)(1 - \alpha) \rceil$-ième plus petit score $\hat q$. Le "+1" compense le fait qu'on a un nombre fini de scores. L'intervalle vaut :
  $$C_t = \big[ \hat y_t - \hat q, \; \hat y_t + \hat q \big], \qquad P(y_t \in C_t) \geq 1 - \alpha$$
  La garantie vaut si les données sont échangeables.
- **Conforme fixe** (*split conformal*) : $\hat q$ est calculé une fois sur l'échantillon de calibration et gardé pour tout le test.
- **Conforme glissant** : $\hat q$ est recalculé à chaque date avec les 200 derniers scores observés, ce qui suit les changements de régime.
- **ACI** (*adaptive conformal inference*, Gibbs et Candès, 2021) : le niveau d'erreur $\alpha_t$ est ajusté après chaque observation. Si l'intervalle a couvert, $\alpha_t$ augmente (intervalle plus étroit), sinon il baisse (intervalle plus large). Avec $\text{err}_t = \mathbb{1}\{ y_t \notin C_t \}$ :
  $$\alpha_{t+1} = \alpha_t + \gamma \big( \alpha - \text{err}_t \big)$$
  où $\gamma$ est le pas d'adaptation (ici 0.005). Une valeur plus grande réagit plus vite mais rend les intervalles plus instables.
- **Garantie de couverture à long terme d'ACI** : quelle que soit la loi des données, même avec une rupture, la couverture moyenne converge vers $1 - \alpha$.
  $$\Big| \frac{1}{T} \sum_{t=1}^{T} \text{err}_t - \alpha \Big| \leq \frac{\max(\alpha_1, 1 - \alpha_1) + \gamma}{\gamma \, T}$$
- **Intervalle infini** : si $\alpha_t \leq 0$, le quantile demandé n'existe pas et l'intervalle devient $(-\infty, +\infty)$. Le code garde ces bornes infinies dans les intervalles, et ne les remplace par une grande valeur que pour calculer largeurs et scores.
- **Régression quantile** : régression qui estime un quantile conditionnel de $y$ (par exemple le quantile à 5 %) au lieu de la moyenne. Elle minimise la perte "pinball" de niveau $\tau$ :
  $$\rho_\tau(u) = u \big( \tau - \mathbb{1}\{ u < 0 \} \big)$$
- **CQR** (*conformalized quantile regression*, Romano et al., 2019) : on estime deux régressions quantiles, $\hat q_{lo}(x)$ au niveau $\alpha / 2$ et $\hat q_{hi}(x)$ au niveau $1 - \alpha / 2$ (ici par gradient boosting), puis on corrige l'intervalle avec un quantile conforme $Q$ des scores de calibration. L'intervalle s'adapte à la volatilité et n'est pas forcément symétrique.
  $$s_i = \max\big( \hat q_{lo}(x_i) - y_i, \; y_i - \hat q_{hi}(x_i) \big), \qquad C_t = \big[ \hat q_{lo}(x_t) - Q, \; \hat q_{hi}(x_t) + Q \big]$$
- **CQR + ACI** : CQR avec l'ajustement en ligne du niveau $\alpha_t$ décrit plus haut.
- **Gradient boosting** : on ajoute des petits arbres un par un, chacun corrigeant les erreurs du précédent, avec un pas $\nu$.
  $$\hat f_t = \hat f_{t-1} + \nu \, h_t$$
- **Réestimation du modèle de base** : refaire l'estimation du modèle au fil du temps. Elle n'est pas faite ici.
- **Horizon** : nombre de pas à l'avance de la prévision. Ici un seul pas, car pour un horizon plus long les erreurs successives sont corrélées et il faut adapter la méthode.
- **EnbPI** : méthode de prévision conforme pour séries temporelles qui utilise un ensemble de modèles entraînés sur des échantillons bootstrap et des résidus hors échantillon.
- **Variantes pondérées, fenêtre adaptative** : variantes qui donnent plus de poids aux scores récents ou qui choisissent automatiquement la taille de la fenêtre.

### Évaluation

- **Couverture par régime** : couverture calculée séparément sur les dates calmes et sur les dates agitées. Elle montre si la garantie marginale cache des écarts selon le régime.
- **Largeur** : largeur moyenne des intervalles. À couverture égale, un intervalle plus étroit est plus informatif.
  $$\text{largeur} = \frac{1}{T} \sum_{t=1}^{T} (u_t - l_t)$$
- **Score d'intervalle** (Gneiting et Raftery, 2007) : mesure qui combine largeur et couverture, car elle ajoute une pénalité proportionnelle à l'écart quand la vraie valeur sort de l'intervalle. Plus il est bas, mieux c'est.
  $$\text{IS}_\alpha = (u_t - l_t) + \frac{2}{\alpha} (l_t - y_t) \, \mathbb{1}\{ y_t < l_t \} + \frac{2}{\alpha} (y_t - u_t) \, \mathbb{1}\{ y_t > u_t \}$$
- **Couverture glissante** : couverture calculée sur une fenêtre de 200 observations qui avance dans le temps. Elle montre à quelles périodes la couverture se dégrade.
- **Pire écart sur 200 observations** : plus grand écart, sur toutes les fenêtres, entre la couverture glissante et la couverture visée.
  $$\max_t \Big| \text{couv}^{(200)}_t - (1 - \alpha) \Big|$$
- **Part infinie** : fréquence des dates où l'intervalle est infini.
- **Sous-couverture** : couverture inférieure au niveau visé, ce qui veut dire que les intervalles sont trop étroits.
