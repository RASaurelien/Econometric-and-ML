## Définition

Notations : $x$ est l'état (kilométrage depuis le dernier remplacement, discrétisé en 90 niveaux), $a \in \{0, 1\}$ l'action (0 garder le moteur, 1 le remplacer), $\beta = 0.95$ le facteur d'escompte, $\theta = (\theta_1, RC)$ les paramètres de coût à estimer et $\varepsilon_a$ le choc privé associé à l'action $a$. $u(x, a)$ est l'utilité courante, $\pi_k$ la probabilité que le kilométrage augmente de $k \in \{0, 1, 2\}$ niveaux et $\gamma \approx 0.5772$ la constante d'Euler. $N$ est le nombre de bus et $T$ le nombre de périodes.

### Le modèle

- **Choix discret dynamique** : à chaque période, un agent choisit une action parmi un nombre fini, et ce choix change l'état futur. Il compare donc le coût d'aujourd'hui et ses conséquences sur les périodes suivantes.
- **Arrêt optimal** : problème où il faut choisir le bon moment pour une action qui remet le système à zéro. Ici : remplacer le moteur.
- **Modèle de Rust (1987)** : le gestionnaire de flotte Harold Zurcher choisit chaque mois de garder ou de remplacer le moteur d'un bus.
- **Utilité courante** : garder le moteur coûte de plus en plus cher avec le kilométrage, le remplacer coûte $RC$ (*replacement cost*) et remet l'état à zéro.
  $$u(x, 0) = -0.01 \, \theta_1 \, x, \qquad u(x, 1) = -RC$$
- **Renouvellement** : le remplacement remet l'état à zéro, donc l'avenir après un remplacement ne dépend pas de $x$. C'est ce qui rend l'inversion de Hotz et Miller linéaire.
- **Transition de l'état** : sans remplacement, le kilométrage monte de 0, 1 ou 2 niveaux avec des probabilités fixes (0.35, 0.5, 0.15).
  $$P(x' = x + k \mid x, a = 0) = \pi_k$$
- **Choc privé** : composante du coût que le gestionnaire observe mais pas l'économètre (panne imprévue, état réel du moteur).
- **Choc de type valeur extrême** (type I, ou Gumbel) : choix de loi pour $\varepsilon_a$ qui rend les probabilités de choix logistiques, donc calculables sans simulation.
- **Facteur d'escompte** : poids de l'avenir par rapport au présent. Avec $\beta = 0.95$, une unité de coût dans une période vaut 0.95 aujourd'hui.
- **Valeur spécifique à l'action** : valeur d'une action avant de connaître le choc, en tenant compte de l'avenir. Avec $V(x)$ la fonction de valeur :
  $$v(x, a) = u(x, a) + \beta \, E\big[ V(x') \mid x, a \big]$$
- **Fonction de valeur** : coût total espéré actualisé, quand on agit de façon optimale à partir de l'état $x$. Avec des chocs de type valeur extrême, elle a une forme fermée (équation de Bellman) :
  $$V(x) = \gamma + \ln \sum_{a} \exp\big( v(x, a) \big)$$
- **Itération sur la fonction de valeur** : on part d'une fonction $V$ quelconque et on applique l'équation de Bellman en boucle. Comme l'opérateur est une contraction (car $\beta < 1$), les valeurs convergent vers l'unique point fixe.
- **CCP** (*conditional choice probabilities*, probabilités de choix conditionnelles) : probabilité de choisir l'action $a$ dans l'état $x$. Ici, c'est un logit.
  $$P(a \mid x) = \frac{\exp\big( v(x, a) \big)}{\sum_{a'} \exp\big( v(x, a') \big)}$$
- **Taux de remplacement stationnaire** : part des périodes où l'on remplace, à long terme. Avec $\pi^*$ la loi stationnaire de la chaîne des états ($\pi^* = \pi^* \mathcal{P}$, où $\mathcal{P}$ est la matrice de transition induite par les choix) :
  $$\bar p_1 = \sum_x \pi^*(x) \, P(1 \mid x)$$
- **Contrefactuel** : on refait le calcul avec un paramètre modifié ici $RC$, pour voir comment le taux de remplacement réagirait. C'est l'équivalent de la courbe de demande de Rust.

### Estimateurs

- **Vraisemblance** : probabilité des choix observés pour des valeurs de $\theta$. On maximise son logarithme.
  $$\ln L(\theta) = \sum_{i=1}^{N} \sum_{t=1}^{T} \ln P(a_{it} \mid x_{it}; \theta)$$
- **Estimation en deux étapes** : on estime d'abord les probabilités de transition $\pi_k$ (par fréquences), puis les paramètres de coût en les prenant comme donnés. C'est pour cela que les écarts-types de NFXP ignorent l'incertitude de la première étape.
- **NFXP** (*nested fixed point*, point fixe imbriqué) : maximum de vraisemblance où, à chaque valeur essayée de $\theta$, on résout le problème dynamique (itération sur la fonction de valeur) pour calculer les CCP. Précis mais lent.
- **Hessien numérique** : matrice des dérivées secondes de $\ln L$ calculée par différences finies. Les écarts-types sont la racine des éléments diagonaux de :
  $$\widehat{\text{Var}}(\hat\theta) = \Big( -\frac{\partial^2 \ln L}{\partial \theta \, \partial \theta^\top} \Big)^{-1}$$
- **Inversion de Hotz et Miller** (1993) : avec des chocs de type valeur extrême, les CCP et les valeurs sont reliées par une formule inversible, ce qui permet d'écrire la valeur à partir des CCP estimées sans résoudre le modèle.
  $$V(x) = v(x, a) - \ln P(a \mid x) + \gamma, \qquad \ln\frac{P(0 \mid x)}{P(1 \mid x)} = v(x, 0) - v(x, 1)$$
  Avec le renouvellement, le second membre devient linéaire en $(\theta_1, RC)$ pour des CCP estimées, donc une régression pondérée suffit.
- **Logit polynomial** : estimation de la probabilité de remplacement par état avec un logit dont l'indice est un polynôme en $x$.
  $$\ln\frac{p_1(x)}{1 - p_1(x)} = \sum_{k=0}^{K} a_k x^k$$
- **Extrapolation** : prédire pour des états peu ou pas observés. Un polynôme extrapole mal, d'où le biais de l'inversion sur $RC$.
- **Régression pondérée** : régression où chaque observation compte plus ou moins, ici selon la précision avec laquelle les CCP sont estimées.
- **NPL** (*nested pseudo likelihood*, Aguirregabiria et Mira, 2002) : on alterne deux étapes. Avec des CCP données $P$, on calcule la valeur de continuation en résolvant un système linéaire (sans itération sur la fonction de valeur) :
  $$V_P = \big( I - \beta F_P \big)^{-1} \sum_a P_a \odot \big( u_a + \gamma - \ln P_a \big)$$
  où $F_P = \sum_a \text{diag}(P_a) F_a$ et $F_a$ est la matrice de transition sous l'action $a$. On maximise ensuite la pseudo-vraisemblance en $\theta$, puis on met à jour les CCP avec ce $\hat\theta$ :
  $$P^{(m+1)} = \Psi\big( \hat\theta^{(m)}, P^{(m)} \big)$$
  où $\Psi$ est l'application qui transforme des CCP et des paramètres en nouvelles CCP (ici 3 itérations).
- **Pseudo-vraisemblance** : vraisemblance où les CCP du second membre sont fixées à une valeur donnée (celles de l'étape précédente) au lieu d'être recalculées à partir de $\theta$.
- **Équivalence asymptotique** : les estimateurs NPL, pour un nombre d'itérations suffisant, ont la même distribution limite que le maximum de vraisemblance (NFXP).
- **Bootstrap sur les bus** : on tire $N$ bus avec remise dans l'échantillon, on réestime, et on répète $B$ fois. L'écart-type des $B$ estimations sert d'écart-type de l'estimateur. On tire des bus entiers, pas des périodes, parce que les périodes d'un même bus sont dépendantes.

### Simulations et évaluation

- **Monte Carlo** : on simule $R$ jeux de données à partir du modèle avec des paramètres vrais fixés, puis on regarde le comportement des estimateurs sur les $R$ répétitions.
- **Biais** : écart moyen entre l'estimation et la vraie valeur.
  $$\text{biais} = \frac{1}{R} \sum_{r=1}^{R} \hat\theta_r - \theta$$
- **RMSE** (*root mean squared error*) : erreur totale, qui mélange biais et dispersion.
  $$\text{RMSE} = \sqrt{\frac{1}{R} \sum_{r=1}^{R} (\hat\theta_r - \theta)^2}$$
- **Graine fixée** : valeur qui fixe l'aléa d'une simulation, pour retrouver les mêmes résultats en relançant le code.
- **Logit sans pénalité** : régression logistique sans terme de régularisation (`penalty=None` dans scikit-learn).
- **Écart quadratique pondéré par les visites** : distance entre deux probabilités de remplacement, qui compte davantage les états souvent visités.
  $$\sum_x w(x) \big( \hat p_1(x) - p_1(x) \big)^2, \qquad w(x) = \frac{\text{visites de } x}{\text{visites totales}}$$

### Apprentissage par renforcement

- **Apprentissage par renforcement** (RL, *reinforcement learning*) : un agent apprend à bien agir en interagissant avec un environnement, guidé par une récompense (ici l'opposé du coût).
- **Q-learning tabulaire** : l'agent garde un tableau $Q(x, a)$ de la valeur de chaque action dans chaque état, et le corrige après chaque pas avec le résultat observé (Watkins et Dayan, 1992).
  $$Q(x, a) \leftarrow Q(x, a) + \alpha \Big[ r + \beta \max_{a'} Q(x', a') - Q(x, a) \Big]$$
  où $r$ est la récompense du pas (ici l'utilité courante plus le choc observé) et $\alpha$ le taux d'apprentissage.
- **Taux d'apprentissage** : poids donné à l'information nouvelle dans la mise à jour, $\alpha \in (0, 1]$.
- **Exploration** : l'agent essaie parfois des actions au hasard (par exemple avec une probabilité $\epsilon$) pour visiter tous les états. Sans elle, il peut rester coincé sur de mauvaises actions.
- **Solveur** : ici le Q-learning résout un problème dont les paramètres sont connus. Il ne les estime pas à partir de données.
- **RL inverse** (IRL) : à l'inverse du RL, on observe des trajectoires d'un agent et on cherche la fonction de récompense qui les explique. Estimer $(\theta_1, RC)$ est de ce type.
- **IRL à entropie maximale** (Ziebart et al., 2008) : modèle d'IRL où la probabilité d'une trajectoire est proportionnelle à l'exponentielle de sa récompense. Le choix logit du modèle de Rust en est un cas particulier.
- **DQN** (*deep Q-network*) : Q-learning où le tableau $Q(x, a)$ est remplacé par un réseau de neurones, utile quand l'espace d'états est continu et ne peut plus être tabulé.
- **Hétérogénéité non observée** : les bus diffèrent par des caractéristiques que l'économètre ne voit pas, ce qui est absent ici.
- **Choc autocorrélé** : le choc d'une période dépend de celui de la précédente, ce que le modèle exclut ici (les chocs sont indépendants entre périodes).
