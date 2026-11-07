import numpy as np

# Toutes les fonctions renvoient la matrice de covariance de beta (k x k).
# X est (n, k), e les residus (n,), bread = (X'X)^-1


# Sandwich de White. Le levier h sert pour HC2 et HC3.
def hc(X, e, bread, variante="HC1"):
    n, k = X.shape
    h = np.einsum("ij,jk,ik->i", X, bread, X)
    if variante == "HC0":
        w = e ** 2
    elif variante == "HC1":
        w = e ** 2 * n / (n - k)
    elif variante == "HC2":
        w = e ** 2 / (1 - h)
    elif variante == "HC3":
        w = e ** 2 / (1 - h) ** 2
    else:
        raise ValueError("variante inconnue : " + variante)
    meat = (X * w[:, None]).T @ X
    return bread @ meat @ bread


# Newey-West avec poids de Bartlett, sans correction de petit echantillon.
def newey_west(X, e, bread, lags):
    u = X * e[:, None]
    meat = u.T @ u
    for l in range(1, lags + 1):
        poids = 1 - l / (lags + 1)
        gamma = u[l:].T @ u[:-l]
        meat += poids * (gamma + gamma.T)
    return bread @ meat @ bread


# Ecarts-types clusterises, avec la correction G/(G-1) * (n-1)/(n-k) de Stata.
def cluster(X, e, bread, groupes):
    n, k = X.shape
    u = X * e[:, None]
    ids = np.unique(groupes)
    meat = np.zeros((k, k))
    for g in ids:
        s = u[groupes == g].sum(axis=0)
        meat += np.outer(s, s)
    G = len(ids)
    corr = G / (G - 1) * (n - 1) / (n - k)
    return corr * bread @ meat @ bread
