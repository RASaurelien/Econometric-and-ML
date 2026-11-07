import numpy as np
from scipy import stats

from .lisseurs import poids_locaux


# Resultat du modele partiellement lineaire. g(grille) renvoie la partie non parametrique.
class ResultatRobinson:
    def __init__(self, beta, v, noms, h, z, partie_np):
        self.beta, self.cov, self.noms, self.h = beta, v, noms, h
        self.z, self.partie_np = z, partie_np
        self.se = np.sqrt(np.diag(v))
        self.t = beta / self.se
        self.p = 2 * stats.norm.sf(np.abs(self.t))

    def g(self, grille, noyau="gaussien", degre=1):
        L = poids_locaux(self.z, grille, self.h, noyau, degre)
        return L @ self.partie_np

    def __str__(self):
        lignes = [f"{'':>12}{'coef':>12}{'se':>12}{'z':>10}{'p':>10}"]
        for i, nom in enumerate(self.noms):
            lignes.append(f"{nom:>12}{self.beta[i]:>12.4f}{self.se[i]:>12.4f}{self.t[i]:>10.2f}{self.p[i]:>10.3f}")
        lignes.append(f"fenetre h = {self.h:.4f}, n = {len(self.z)}")
        return "\n".join(lignes)


# Modele partiellement lineaire y = X b + g(z) + e (Robinson, 1988). On retire de y et de X leur
# esperance conditionnelle a z, estimee par polynome local, puis on fait les MCO sur les residus.
# b est convergent a vitesse racine de n, g reste libre. Pas de constante dans X : elle est dans g.
# Par defaut h = ecart-type de z * n^(-1/3), plus petit que la fenetre usuelle (biais de lissage negligeable).
# Variance robuste (White) sur les residus.
def robinson(y, X, z, h=None, noyau="gaussien", degre=1, noms=None):
    y, z = np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    X = np.asarray(X, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    n = len(y)
    h = np.std(z) * n ** (-1 / 3) if h is None else h
    L = poids_locaux(z, z, h, noyau, degre)
    ey, eX = y - L @ y, X - L @ X
    bread = np.linalg.inv(eX.T @ eX)
    beta = bread @ eX.T @ ey
    u = ey - eX @ beta
    v = bread @ ((eX * u[:, None]).T @ (eX * u[:, None])) @ bread
    noms = list(noms) if noms is not None else [f"x{i}" for i in range(1, X.shape[1] + 1)]
    return ResultatRobinson(beta, v, noms, h, z, L @ (y - X @ beta))
