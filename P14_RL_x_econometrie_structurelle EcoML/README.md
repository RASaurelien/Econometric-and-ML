# Choix discrets dynamiques : le modèle de remplacement de moteurs de Rust (1987)

Ce projet reprend le modèle d'arrêt optimal de Rust (1987) : chaque période, un gestionnaire de flotte décide de garder ou de remplacer le moteur d'un bus, en fonction du kilométrage et d'un choc privé. On simule des données à partir du modèle, puis on compare trois façons d'estimer les paramètres de coût et une façon de résoudre le problème avec de l'apprentissage par renforcement.

Le sujet est inspiré du livre de Greene (Econometric Analysis) et des articles listés dans la section Références.

## Le modèle

L'état est le kilométrage depuis le dernier remplacement, discrétisé en 90 niveaux. Garder le moteur coûte `theta1 * 0.01 * x` par période, le remplacer coûte `RC` et remet le kilométrage à zéro. Le kilométrage augmente de 0, 1 ou 2 niveaux avec des probabilités fixes. Le choc est de type valeur extrême, donc les probabilités de choix sont logistiques, et le facteur d'escompte est `beta = 0.95`.

Les valeurs vraies (`theta1 = 2.5`, `RC = 10`, probabilités d'incrément 0.35, 0.5 et 0.15) sont des choix pour la simulation, ce ne sont pas les estimations de Rust sur les vraies données.

## Estimateurs

| fichier | contenu |
|---|---|
| `bus_model.py` | modèle, itération sur la fonction de valeur, simulation, taux de remplacement stationnaire |
| `nfxp.py` | maximum de vraisemblance avec point fixe imbriqué (NFXP) |
| `ccp.py` | estimateurs à partir des probabilités de choix conditionnelles (CCP) |
| `rl_agent.py` | Q-learning tabulaire dans le simulateur du modèle |
| `main.py` | un run complet avec graphiques, ou un Monte Carlo (`--mc`) |
| `test_rust.py` | vérifications du code sur des cas à solution connue |

- NFXP : la vraisemblance est maximisée sur `(theta1, RC)`, et le problème de programmation dynamique est résolu à chaque évaluation. Les écarts-types viennent d'un hessien numérique.
- CCP, inversion de Hotz et Miller : on estime la probabilité de remplacement par état avec un logit polynomial, puis `ln(p0 / p1)` s'écrit comme une fonction linéaire de `(theta1, RC)` grâce au renouvellement (le remplacement remet l'état à zéro). L'estimation est donc une régression pondérée, sans résoudre le modèle.
- CCP, NPL : les CCP du premier stade servent à calculer la valeur de continuation avec un simple système linéaire, puis on maximise la pseudo-vraisemblance et on met les CCP à jour (3 itérations). C'est l'algorithme de Aguirregabiria et Mira (2002).
- Q-learning : un agent apprend `Q(x, a)` en interagissant avec le simulateur, en observant les chocs comme dans le modèle logit. On compare sa probabilité de remplacement à celle de la programmation dynamique, avec les mêmes paramètres.

Les écarts-types des deux estimateurs CCP sont calculés par bootstrap sur les bus. Les écarts-types de NFXP ignorent l'incertitude de la première étape (estimation des transitions), comme dans la version en deux étapes de Rust.

## Résultats observés pendant le développement

Monte Carlo avec 100 répétitions, 100 bus sur 120 périodes (`python main.py --mc 100`) :

| méthode | biais theta1 | RMSE theta1 | biais RC | RMSE RC | temps moyen |
|---|---|---|---|---|---|
| NFXP | 0.039 | 0.175 | 0.088 | 0.498 | 0.39 s |
| CCP, inversion | 0.030 | 0.178 | -0.769 | 0.872 | 0.004 s |
| CCP, NPL | 0.038 | 0.175 | 0.087 | 0.498 | 0.013 s |

- L'inversion de Hotz et Miller est biaisée sur `RC`. Un diagnostic a montré que le biais vient du terme de valeur future, qui utilise `ln p1` estimé aux états voisins : en le remplaçant par la vraie valeur, le biais disparaît. Le logit polynomial extrapole mal dans les états peu visités (voir le graphique `results/rust.png`).
- L'algorithme NPL corrige ce problème, car les CCP sont recalculées dans tous les états à partir du modèle. Ses résultats sont presque identiques à ceux de NFXP, comme le prédit la théorie (les estimateurs de la suite NPL ont la même distribution asymptotique que le maximum de vraisemblance), pour un temps de calcul environ 30 fois plus faible.
- Dans un run de test, le Q-learning (400 000 pas) donne des probabilités de remplacement proches de la programmation dynamique, avec un écart quadratique pondéré par les visites d'environ 0.013.

Ces chiffres viennent de mes exécutions pendant l'écriture du code, avec une graine fixée. Ils sont à refaire chez soi avant de les citer.

## Contrefactuel

Le graphique de droite de `results/rust.png` reprend l'idée de la courbe de demande de Rust : le taux de remplacement stationnaire en fonction du coût de remplacement `RC`, calculé avec les paramètres vrais et avec les estimations. Comme `RC` est la variable qu'on fait varier, seule l'estimation de `theta1` entre dans la différence entre les courbes.

## Utilisation

```
pip install numpy scipy pandas scikit-learn matplotlib
python test_rust.py
python main.py
python main.py --buses 200 --periods 150 --rl_steps 1000000
python main.py --mc 100
```

Il faut scikit-learn 1.2 ou plus récent (le logit sans pénalité utilise `penalty=None`).

## Limites

- Les données sont simulées. Le code n'a pas été appliqué aux vraies données de Rust. Elles sont disponibles avec le logiciel NFXP, et un dépôt de réplication les décrit (https://rust-data.readthedocs.io/en/latest/replicate_descriptives.html).
- Le Q-learning est ici un solveur dans un modèle connu : l'agent ne retrouve pas les paramètres à partir des données. La relation avec l'apprentissage par renforcement inverse tient à ce que le modèle de choix discret dynamique avec choc de type valeur extrême est un modèle d'entropie maximale. Une vraie approche inverse (estimer la fonction de récompense à partir de trajectoires) n'est pas implémentée.
- Pas de version à réseau de neurones de l'agent (un DQN serait utile pour un espace d'états continu).
- Le modèle est volontairement simple : un seul état, pas d'hétérogénéité non observée et pas de choc auto-corrélé.
- Le test du Q-learning et les comparaisons Monte Carlo utilisent des graines fixes, la vitesse de convergence du Q-learning dépend du taux d'apprentissage et de l'exploration, réglés à la main.

## Références

- Rust, J. (1987), « Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher », *Econometrica* 55(5), 999-1033. https://www.econometricsociety.org/publications/econometrica/1987/09/01/optimal-replacement-gmc-bus-engines-empirical-model-harold
- Hotz, V. J. et Miller, R. (1993), « Conditional Choice Probabilities and the Estimation of Dynamic Models », *Review of Economic Studies* 60(3), 497-529. https://ideas.repec.org/a/oup/restud/v60y1993i3p497-529..html
- Aguirregabiria, V. et Mira, P. (2002), « Swapping the Nested Fixed Point Algorithm: A Class of Estimators for Discrete Markov Decision Models », *Econometrica* 70(4), 1519-1543. https://doi.org/10.1111/1468-0262.00340
- Notes de cours qui présentent le modèle de Rust comme un modèle d'apprentissage par renforcement inverse (source pédagogique, pas un article) : https://colleen.quarto.pub/econ060-engineering-economics/chapters/notes-20.html
- Watkins, C. et Dayan, P. (1992), « Q-learning », *Machine Learning* ; Sutton, R. et Barto, A. (2018), *Reinforcement Learning: An Introduction* (Q-learning)
- Ziebart, B. et al. (2008), « Maximum Entropy Inverse Reinforcement Learning », AAAI (lien entre choix logit et IRL à entropie maximale)
