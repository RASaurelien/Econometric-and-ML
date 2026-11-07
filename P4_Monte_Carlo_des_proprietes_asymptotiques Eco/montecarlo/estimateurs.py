import numpy as np


# Poids du sandwich pour chaque variante de White.
def poids_hc(variante, h, n, k):
    if variante == "HC0":
        return np.ones_like(h)
    if variante == "HC1":
        return np.full_like(h, n / (n - k))
    if variante == "HC2":
        return 1 / (1 - h)
    if variante == "HC3":
        return 1 / (1 - h) ** 2
    raise ValueError("variante inconnue : " + variante)


# MCO sur plusieurs echantillons a la fois. X a la forme (..., n, k) et y (..., n),
# les dimensions de tete sont les replications (et eventuellement les tirages bootstrap).
class MCOBatch:
    def __init__(self, y, X):
        self.n, self.k = X.shape[-2:]
        self.inv = np.linalg.inv(np.einsum("...ni,...nj->...ij", X, X))
        self.A = self.inv @ np.swapaxes(X, -1, -2)
        self.beta = np.einsum("...kn,...n->...k", self.A, y)
        self.e = y - np.einsum("...nk,...k->...n", X, self.beta)
        self.h = np.einsum("...nk,...kn->...n", X, self.A)

    # Ecart-type de chaque coefficient, classique ou sandwich de White.
    def se(self, variante="classique"):
        if variante == "classique":
            s2 = (self.e ** 2).sum(axis=-1) / (self.n - self.k)
            return np.sqrt(s2[..., None] * np.einsum("...ii->...i", self.inv))
        w = poids_hc(variante, self.h, self.n, self.k) * self.e ** 2
        return np.sqrt(np.einsum("...kn,...n,...kn->...k", self.A, w, self.A))


# Variables instrumentales avec un seul regresseur endogene et sans constante.
# Toutes les statistiques ne dependent que de six produits scalaires.
class IVBatch:
    def __init__(self, y, x, z):
        self.n, self.m = z.shape[1:]
        zz = np.linalg.inv(np.einsum("rnm,rnl->rml", z, z))
        zx, zy = np.einsum("rnm,rn->rm", z, x), np.einsum("rnm,rn->rm", z, y)
        q = lambda a, b: np.einsum("rm,rml,rl->r", a, zz, b)
        self.pxx, self.pxy, self.pyy = q(zx, zx), q(zx, zy), q(zy, zy)
        self.axx, self.axy, self.ayy = (x * x).sum(1), (x * y).sum(1), (y * y).sum(1)

    # Estimateur de la classe k : kappa = 1 donne les 2SLS, kappa = kappa du LIML donne le LIML.
    # Renvoie le coefficient et son ecart-type.
    def estimer(self, kappa):
        mxx, mxy = self.axx - self.pxx, self.axy - self.pxy
        d = self.axx - kappa * mxx
        beta = (self.axy - kappa * mxy) / d
        s2 = (self.ayy - 2 * beta * self.axy + beta ** 2 * self.axx) / self.n
        return beta, np.sqrt(s2 / d)

    def deux_smc(self):
        return self.estimer(1.0)

    # Le kappa du LIML est la plus petite valeur propre de (Y'M Y)^-1 Y'Y, avec Y = [y x].
    def kappa_liml(self):
        S = np.stack([np.stack([self.ayy, self.axy], -1), np.stack([self.axy, self.axx], -1)], -2)
        P = np.stack([np.stack([self.pyy, self.pxy], -1), np.stack([self.pxy, self.pxx], -1)], -2)
        return np.linalg.eigvals(np.linalg.solve(S - P, S)).real.min(axis=1)

    def liml(self):
        return self.estimer(self.kappa_liml())

    # F de la premiere etape (instruments tous nuls).
    def f_premiere_etape(self):
        return (self.pxx / self.m) / ((self.axx - self.pxx) / (self.n - self.m))

    # Statistique d'Anderson-Rubin pour H0 : beta = b0. Loi F(m, n-m) sous H0 avec erreurs normales,
    # quelle que soit la force des instruments.
    def anderson_rubin(self, b0):
        mxx, mxy, myy = self.axx - self.pxx, self.axy - self.pxy, self.ayy - self.pyy
        num = (self.pyy - 2 * b0 * self.pxy + b0 ** 2 * self.pxx) / self.m
        den = (myy - 2 * b0 * mxy + b0 ** 2 * mxx) / (self.n - self.m)
        return num / den
