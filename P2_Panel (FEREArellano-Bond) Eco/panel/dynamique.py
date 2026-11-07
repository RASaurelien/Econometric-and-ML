import numpy as np
from scipy import stats

from .outils import coder, preparer


# Resultat d'un Arellano-Bond. Les p-values utilisent la loi normale.
class ResultatDyn:
    def __init__(self, beta, v, noms, N, n_eq, n_inst):
        self.beta = beta
        self.cov = v
        self.noms = noms
        self.N = N
        self.n_eq = n_eq
        self.n_inst = n_inst
        self.k = len(beta)
        self.se = np.sqrt(np.diag(v))
        self.t = beta / self.se
        self.p = 2 * stats.norm.sf(np.abs(self.t))

    def __str__(self):
        lignes = [f"{'':>12}{'coef':>12}{'se':>12}{'z':>10}{'p':>10}"]
        for i, nom in enumerate(self.noms):
            lignes.append(
                f"{nom:>12}{self.beta[i]:>12.4f}{self.se[i]:>12.4f}"
                f"{self.t[i]:>10.2f}{self.p[i]:>10.3f}"
            )
        lignes.append(f"individus = {self.N}, instruments = {self.n_inst}")
        lignes.append("AR(1) : z = %.2f, p = %.3f" % self.ar1)
        lignes.append("AR(2) : z = %.2f, p = %.3f" % self.ar2)
        if self.hansen is not None:
            lignes.append("Hansen : chi2 = %.2f, ddl = %d, p = %.3f" % self.hansen)
        return "\n".join(lignes)


# Donnees en forme (N, T) et (N, T, q), puis variables en differences premieres.
# Pour l'equation en t, la variable dependante est dy_t, les regresseurs sont
# dy_{t-1}, ..., dy_{t-lags} et dx_t. Les equations vont de t = lags+1 a T-1.
def construire(Y, XX, lags):
    d = np.diff(Y, axis=1)
    dX = np.diff(XX, axis=1)
    T = Y.shape[1]
    dep = d[:, lags:]
    retards = [d[:, lags - j : T - 1 - j] for j in range(1, lags + 1)]
    R = np.stack(retards + [dX[:, lags:, q] for q in range(XX.shape[2])], axis=2)
    return dep, R, dX[:, lags:, :]


# Instruments de type GMM : pour l'equation en t, on utilise y_0, ..., y_{t-2}.
# Avec collapse, une seule colonne par distance au lieu d'une par (t, s).
def instruments_gmm(Y, lags, collapse):
    N, T = Y.shape
    n_eq = T - 1 - lags
    largeur = T - 2 if collapse else sum(lags + r for r in range(n_eq))
    Z = np.zeros((N, n_eq, largeur))
    debut = 0
    for r in range(n_eq):
        t = lags + 1 + r
        for s in range(t - 1):
            col = t - s - 2 if collapse else debut + s
            Z[:, r, col] = Y[:, s]
        debut += t - 1
    return Z


# Matrice H des erreurs en differences : 2 sur la diagonale, -1 a cote.
def matrice_h(n_eq):
    return 2 * np.eye(n_eq) - np.eye(n_eq, k=1) - np.eye(n_eq, k=-1)


def etape_gmm(XZ, Zy, W):
    return np.linalg.solve(XZ @ W @ XZ.T, XZ @ W @ Zy)


# Scores par individu g_i = Z_i' e_i, de forme (N, m).
def scores(Z, e):
    return np.einsum("irm,ir->im", Z, e)


# Test d'autocorrelation d'ordre m sur les residus en differences.
# Le terme d2 et d3 corrigent pour l'estimation de beta.
def test_ar(e, R, Z, XZ, W, V, m):
    if m >= e.shape[1]:
        return np.nan, np.nan
    el = np.zeros_like(e)
    el[:, m:] = e[:, :-m]
    w = (el * e).sum(axis=1)
    a = np.einsum("ir,irk->k", el, R)
    B = np.linalg.solve(XZ @ W @ XZ.T, XZ @ W)
    d1 = (w ** 2).sum()
    d2 = -2 * a @ B @ (scores(Z, e) * w[:, None]).sum(axis=0)
    d3 = a @ V @ a
    z = w.sum() / np.sqrt(d1 + d2 + d3)
    return z, 2 * stats.norm.sf(abs(z))


# Arellano-Bond (GMM en differences) pour y_it = rho_1 y_{i,t-1} + ... + x_it b + u_i + e_it.
# y et ids ont une ligne par observation, dans l'ordre du temps pour chaque individu.
# Le panel doit etre equilibre. Les x sont supposes strictement exogenes.
# Etape 1 : poids supposant des erreurs homoscedastiques (matrice H).
# Etape 2 : poids = inverse de la variance des moments de l'etape 1.
# Windmeijer : la matrice D mesure comment b2 bouge quand S depend de b1,
# elle sert a corriger la variance de l'etape 2, trop petite sinon.
def arellano_bond(y, X, ids, lags=1, deux_etapes=True, collapse=False, noms=None):
    y = np.asarray(y, dtype=float)
    X = np.zeros((len(y), 0)) if X is None else preparer(X)
    c, N = coder(ids)
    ordre = np.argsort(c, kind="stable")
    y, X = y[ordre], X[ordre]
    T = len(y) // N
    if T * N != len(y) or not np.all(np.bincount(c) == T):
        raise ValueError("panel desequilibre")
    q = X.shape[1]
    Y = y.reshape(N, T)
    dep, R, dXe = construire(Y, X.reshape(N, T, q), lags)
    Z = np.concatenate([instruments_gmm(Y, lags, collapse), dXe], axis=2)
    n_eq, m = dep.shape[1], Z.shape[2]

    H = matrice_h(n_eq)
    XZ = np.einsum("irk,irm->km", R, Z)
    Zy = np.einsum("irm,ir->m", Z, dep)
    W1 = np.linalg.pinv(np.einsum("ira,rs,isb->ab", Z, H, Z, optimize=True))

    b1 = etape_gmm(XZ, Zy, W1)
    e1 = dep - np.einsum("irk,k->ir", R, b1)
    g1 = scores(Z, e1)
    S1 = g1.T @ g1
    A1 = np.linalg.inv(XZ @ W1 @ XZ.T)
    V1 = A1 @ XZ @ W1 @ S1 @ W1 @ XZ.T @ A1

    nom = list(noms) if noms is not None else [f"x{i}" for i in range(1, q + 1)]
    noms_all = [f"L{j}.y" for j in range(1, lags + 1)] + nom

    if not deux_etapes:
        res = ResultatDyn(b1, V1, noms_all, N, n_eq, m)
        res.ar1 = test_ar(e1, R, Z, XZ, W1, V1, 1)
        res.ar2 = test_ar(e1, R, Z, XZ, W1, V1, 2)
        res.hansen = None
        return res

    W2 = np.linalg.pinv(S1)
    b2 = etape_gmm(XZ, Zy, W2)
    e2 = dep - np.einsum("irk,k->ir", R, b2)
    A2 = np.linalg.inv(XZ @ W2 @ XZ.T)
    V2 = A2

    g2 = scores(Z, e2).sum(axis=0)
    D = np.zeros((len(b2), len(b2)))
    for j in range(len(b2)):
        h = scores(Z, R[:, :, j])
        dS = h.T @ g1 + g1.T @ h
        D[:, j] = A2 @ XZ @ W2 @ dS @ W2 @ g2
    Vw = V2 + D @ V2 + V2 @ D.T + D @ V1 @ D.T

    res = ResultatDyn(b2, Vw, noms_all, N, n_eq, m)
    res.ar1 = test_ar(e2, R, Z, XZ, W2, V2, 1)
    res.ar2 = test_ar(e2, R, Z, XZ, W2, V2, 2)
    stat = g2 @ W2 @ g2
    ddl = m - len(b2)
    res.hansen = (stat, ddl, stats.chi2.sf(stat, ddl)) if ddl > 0 else None
    res.cov_asymptotique = V2
    res.cov_etape1 = V1
    return res
