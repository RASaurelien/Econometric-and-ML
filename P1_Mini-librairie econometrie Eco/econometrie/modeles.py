import numpy as np
from scipy import stats

from . import covariances as cov


# Resultat d'une estimation. Les p-values utilisent une loi de Student a n-k ddl.
class Resultat:
    def __init__(self, y, X, beta, v, noms):
        self.y = y
        self.X = X
        self.beta = beta
        self.cov = v
        self.noms = noms
        self.n, self.k = X.shape
        self.residus = y - X @ beta
        self.se = np.sqrt(np.diag(v))
        self.t = beta / self.se
        self.p = 2 * stats.t.sf(np.abs(self.t), self.n - self.k)

    # R2 calcule sur les residus bruts, centre seulement s'il y a une constante.
    def r2(self):
        sse = self.residus @ self.residus
        base = self.y - self.y.mean() if np.allclose(self.X[:, 0], 1) else self.y
        return 1 - sse / (base @ base)

    def __str__(self):
        lignes = [f"{'':>10}{'coef':>12}{'se':>12}{'t':>10}{'p':>10}"]
        for i, nom in enumerate(self.noms):
            lignes.append(
                f"{nom:>10}{self.beta[i]:>12.4f}{self.se[i]:>12.4f}"
                f"{self.t[i]:>10.2f}{self.p[i]:>10.3f}"
            )
        lignes.append(f"n = {self.n}, R2 = {self.r2():.3f}")
        return "\n".join(lignes)


def preparer(X):
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def ajouter_constante(X):
    return np.column_stack([np.ones(len(X)), X])


def donner_noms(noms, k, constante):
    if noms is not None:
        return list(noms)
    debut = ["const"] if constante else []
    return debut + [f"x{i}" for i in range(1, k - len(debut) + 1)]


# Choisit l'estimateur de variance selon le nom donne.
def choisir_cov(X, e, bread, methode, lags, groupes):
    n, k = X.shape
    if methode == "classique":
        return e @ e / (n - k) * bread
    if methode.startswith("HC"):
        return cov.hc(X, e, bread, methode)
    if methode == "HAC":
        if lags is None:
            lags = int(4 * (n / 100) ** (2 / 9))
        return cov.newey_west(X, e, bread, lags)
    if methode == "cluster":
        return cov.cluster(X, e, bread, np.asarray(groupes))
    raise ValueError("methode inconnue : " + methode)


# Moindres carres ordinaires.
def mco(y, X, constante=True, cov="classique", lags=None, groupes=None, noms=None):
    y = np.asarray(y, dtype=float)
    X = preparer(X)
    if constante:
        X = ajouter_constante(X)
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    e = y - X @ beta
    v = choisir_cov(X, e, bread, cov, lags, groupes)
    return Resultat(y, X, beta, v, donner_noms(noms, X.shape[1], constante))


# Moindres carres generalises quand les variances des erreurs sont connues.
# On divise chaque ligne par l'ecart-type de l'erreur puis on fait un MCO.
def mcg(y, X, variances, constante=True, noms=None):
    y = np.asarray(y, dtype=float)
    X = preparer(X)
    if constante:
        X = ajouter_constante(X)
    n, k = X.shape
    w = 1 / np.sqrt(np.asarray(variances, dtype=float))
    Xs, ys = X * w[:, None], y * w
    bread = np.linalg.inv(Xs.T @ Xs)
    beta = bread @ Xs.T @ ys
    es = ys - Xs @ beta
    v = es @ es / (n - k) * bread
    return Resultat(y, X, beta, v, donner_noms(noms, k, constante))


# MCG faisables avec heteroscedasticite multiplicative : on estime
# log(e^2) par une regression sur X puis on en tire les variances.
def mcgf(y, X, constante=True, noms=None):
    base = mco(y, X, constante)
    g = np.linalg.lstsq(base.X, np.log(base.residus ** 2), rcond=None)[0]
    variances = np.exp(base.X @ g)
    return mcg(y, X, variances, constante, noms)


# Doubles moindres carres. X contient tous les regresseurs, endogenes compris,
# Z contient les instruments ET les regresseurs exogenes.
def mc2e(y, X, Z, constante=True, noms=None):
    y = np.asarray(y, dtype=float)
    X, Z = preparer(X), preparer(Z)
    if constante:
        X, Z = ajouter_constante(X), ajouter_constante(Z)
    n, k = X.shape
    Xhat = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)
    bread = np.linalg.inv(Xhat.T @ Xhat)
    beta = bread @ Xhat.T @ y
    e = y - X @ beta
    res = Resultat(y, X, beta, e @ e / (n - k) * bread, donner_noms(noms, k, constante))
    res.Z = Z
    return res


# Estimateur GMM pour une matrice de poids W donnee.
def estimer_gmm(y, X, Z, W):
    A = X.T @ Z @ W @ Z.T @ X
    return np.linalg.solve(A, X.T @ Z @ W @ Z.T @ y)


# GMM efficace en deux etapes, robuste a l'heteroscedasticite.
# Etape 1 : poids de la 2SLS. Etape 2 : poids = inverse de la variance des moments.
# La variance finale re-estime S avec les residus de l'etape 2.
def gmm(y, X, Z, constante=True, noms=None):
    y = np.asarray(y, dtype=float)
    X, Z = preparer(X), preparer(Z)
    if constante:
        X, Z = ajouter_constante(X), ajouter_constante(Z)
    n, k = X.shape
    b1 = estimer_gmm(y, X, Z, np.linalg.inv(Z.T @ Z / n))
    e1 = y - X @ b1
    S = (Z * (e1 ** 2)[:, None]).T @ Z / n
    W = np.linalg.inv(S)
    beta = estimer_gmm(y, X, Z, W)
    e = y - X @ beta
    S = (Z * (e ** 2)[:, None]).T @ Z / n
    G = Z.T @ X / n
    v = np.linalg.inv(G.T @ np.linalg.inv(S) @ G) / n
    res = Resultat(y, X, beta, v, donner_noms(noms, k, constante))
    res.Z = Z
    return res
