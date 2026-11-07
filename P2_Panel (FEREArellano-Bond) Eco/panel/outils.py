import numpy as np
from scipy import stats


# Transforme les identifiants en codes 0..N-1 et renvoie aussi N.
def coder(ids):
    labels, c = np.unique(ids, return_inverse=True)
    return c, len(labels)


# Moyenne par individu, une ligne par individu.
def moyennes(M, c, N):
    M2 = M.reshape(len(M), -1)
    somme = np.zeros((N, M2.shape[1]))
    np.add.at(somme, c, M2)
    t = np.bincount(c, minlength=N)
    return (somme / t[:, None]).reshape((N,) + M.shape[1:])


# Transformation within : on retire la moyenne de chaque individu.
def intra(M, c, N):
    return M - moyennes(M, c, N)[c]


def preparer(X):
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


# Resultat d'une estimation en panel. Les p-values suivent une loi de Student a ddl degres.
class Resultat:
    def __init__(self, beta, v, residus, noms, ddl, n, N):
        self.beta = beta
        self.cov = v
        self.residus = residus
        self.noms = noms
        self.ddl = ddl
        self.n = n
        self.N = N
        self.k = len(beta)
        self.rss = residus @ residus
        self.se = np.sqrt(np.diag(v))
        self.t = beta / self.se
        self.p = 2 * stats.t.sf(np.abs(self.t), ddl)

    def __str__(self):
        lignes = [f"{'':>12}{'coef':>12}{'se':>12}{'t':>10}{'p':>10}"]
        for i, nom in enumerate(self.noms):
            lignes.append(
                f"{nom:>12}{self.beta[i]:>12.4f}{self.se[i]:>12.4f}"
                f"{self.t[i]:>10.2f}{self.p[i]:>10.3f}"
            )
        lignes.append(f"n = {self.n}, individus = {self.N}")
        return "\n".join(lignes)
