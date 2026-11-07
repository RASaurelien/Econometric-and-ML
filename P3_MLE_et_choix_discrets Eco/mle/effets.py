import numpy as np
from scipy import stats

from .estimation import jacobien


# Effets marginaux moyens (ou en la moyenne des x) avec ecarts-types par delta method.
# Une colonne qui ne prend que 0 et 1 donne une difference discrete, les autres une derivee.
# Les arguments supplementaires (categorie, quoi) sont passes a res.moyenne.
class EffetsMarginaux:
    def __init__(self, noms, valeurs, v):
        self.noms = noms
        self.valeurs = valeurs
        self.se = np.sqrt(np.diag(v))
        self.z = valeurs / self.se
        self.p = 2 * stats.norm.sf(np.abs(self.z))

    def __str__(self):
        lignes = [f"{'':>16}{'effet':>12}{'se':>12}{'z':>10}{'p':>10}"]
        for i, nom in enumerate(self.noms):
            lignes.append(
                f"{nom:>16}{self.valeurs[i]:>12.4f}{self.se[i]:>12.4f}"
                f"{self.z[i]:>10.2f}{self.p[i]:>10.3f}"
            )
        return "\n".join(lignes)


def effets_marginaux(res, en_moyenne=False, **kw):
    X = res.X
    colonnes = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 0]
    base = X.mean(axis=0, keepdims=True) if en_moyenne else X

    def effets(theta):
        sortie = []
        for j in colonnes:
            haut, bas = base.copy(), base.copy()
            if set(np.unique(X[:, j])) <= {0.0, 1.0}:
                haut[:, j], bas[:, j], pas = 1.0, 0.0, 1.0
            else:
                pas = 2e-5 * max(1.0, np.abs(X[:, j]).mean())
                haut[:, j] += pas / 2
                bas[:, j] -= pas / 2
            sortie.append(np.mean((res.moyenne(theta, haut, **kw) - res.moyenne(theta, bas, **kw)) / pas))
        return np.array(sortie)

    J = jacobien(effets, res.theta)
    return EffetsMarginaux([res.noms_x[j] for j in colonnes], effets(res.theta), J @ res.cov @ J.T)
