import numpy as np

from . import covariances as cov
from .outils import coder, moyennes, intra, preparer, Resultat


def nommer(noms, K):
    return list(noms) if noms is not None else [f"x{i}" for i in range(1, K + 1)]


# Facteur de petit echantillon pour la variance clusterisee : G/(G-1) * (n-1)/(n-k).
def facteur(n, k, G):
    return G / (G - 1) * (n - 1) / (n - k)


# MCO sur des donnees deja transformees. Renvoie beta, sa variance et les residus.
def ols(y, X, c, N, ddl, methode, corr=1.0):
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    e = y - X @ beta
    if methode == "cluster":
        v = cov.cluster(X, e, bread, c, N, corr)
    else:
        v = cov.classique(e, bread, ddl)
    return beta, v, e


# MCO empiles : on ignore la dimension panel, avec une constante.
def mco_empile(y, X, ids, cov="classique", noms=None):
    y, X = np.asarray(y, dtype=float), preparer(X)
    c, N = coder(ids)
    n = len(y)
    Xc = np.column_stack([np.ones(n), X])
    k = Xc.shape[1]
    beta, v, e = ols(y, Xc, c, N, n - k, cov, facteur(n, k, N))
    return Resultat(beta, v, e, ["const"] + nommer(noms, k - 1), n - k, n, N)


# Estimateur between : regression des moyennes individuelles.
def between(y, X, ids, noms=None):
    y, X = np.asarray(y, dtype=float), preparer(X)
    c, N = coder(ids)
    yb = moyennes(y, c, N)
    Xb = np.column_stack([np.ones(N), moyennes(X, c, N)])
    k = Xb.shape[1]
    beta, v, e = ols(yb, Xb, np.arange(N), N, N - k, "classique")
    return Resultat(beta, v, e, ["const"] + nommer(noms, k - 1), N - k, len(y), N)


# Effets fixes : MCO sur les donnees centrees par individu (within).
# Les degres de liberte tiennent compte des N moyennes estimees.
def effets_fixes(y, X, ids, cov="classique", noms=None):
    y, X = np.asarray(y, dtype=float), preparer(X)
    c, N = coder(ids)
    n, K = X.shape
    ddl = n - N - K
    beta, v, e = ols(intra(y, c, N), intra(X, c, N), c, N, ddl, cov, facteur(n, K, N))
    res = Resultat(beta, v, e, nommer(noms, K), ddl, n, N)
    res.effets = moyennes(y, c, N) - moyennes(X, c, N) @ beta
    return res


# Effets aleatoires (Swamy-Arora) : on estime les variances avec le within et
# le between, puis on fait un MCO sur les donnees quasi-centrees.
def effets_aleatoires(y, X, ids, noms=None):
    y, X = np.asarray(y, dtype=float), preparer(X)
    c, N = coder(ids)
    n, K = X.shape
    fe = effets_fixes(y, X, ids)
    be = between(y, X, ids)
    sigma_e2 = fe.rss / fe.ddl
    sigma_b2 = be.rss / be.ddl
    T = np.bincount(c)
    T_harm = N / np.sum(1 / T)
    sigma_u2 = max(sigma_b2 - sigma_e2 / T_harm, 0)
    theta = 1 - np.sqrt(sigma_e2 / (sigma_e2 + T * sigma_u2))
    th = theta[c]
    ys = y - th * moyennes(y, c, N)[c]
    Xs = np.column_stack([1 - th, X - th[:, None] * moyennes(X, c, N)[c]])
    beta, v, e = ols(ys, Xs, c, N, n - K - 1, "classique")
    res = Resultat(beta, v, e, ["const"] + nommer(noms, K), n - K - 1, n, N)
    res.theta, res.sigma_e2, res.sigma_u2 = theta, sigma_e2, sigma_u2
    return res


# Mundlak : MCO empiles avec les moyennes individuelles des regresseurs en plus,
# variance clusterisee. Les coefficients des x sont ceux des effets fixes.
def mundlak(y, X, ids, noms=None):
    y, X = np.asarray(y, dtype=float), preparer(X)
    c, N = coder(ids)
    n, K = X.shape
    Xa = np.column_stack([np.ones(n), X, moyennes(X, c, N)[c]])
    k = Xa.shape[1]
    beta, v, e = ols(y, Xa, c, N, n - k, "cluster", facteur(n, k, N))
    nom = nommer(noms, K)
    res = Resultat(beta, v, e, ["const"] + nom + ["moy_" + s for s in nom], n - k, n, N)
    res.idx_moyennes = list(range(K + 1, 2 * K + 1))
    return res
