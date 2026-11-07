import numpy as np
from scipy import stats
from scipy.optimize import linprog
from scipy.stats import norm

from .noyaux import NOYAUX


# Fonction de perte du quantile : somme de u * (tau - 1{u < 0}).
def perte(u, tau):
    return np.sum(u * (tau - (u < 0)))


# Resolution du probleme de la regression quantile par programmation lineaire.
# Le primal (min somme de rho_tau(y - X b)) a n contraintes. On resout plutot son dual,
#   max y'a   sous X'a = (1 - tau) X'1,   0 <= a <= 1,
# qui n'a que k contraintes. Les coefficients sont les multiplicateurs de ces contraintes.
# Au-dela de 20 000 observations on prend la methode des points interieurs de HiGHS, environ 5 fois
# plus rapide pour la meme solution.
def resoudre(y, X, tau):
    methode = "highs-ipm" if len(y) > 20000 else "highs"
    res = linprog(-y, A_eq=X.T, b_eq=(1 - tau) * X.sum(axis=0), bounds=(0, 1), method=methode)
    if res.status != 0:
        raise RuntimeError("programme lineaire non resolu : " + res.message)
    return -res.eqlin.marginals


# Fenetre de Hall et Sheather (1988) pour estimer la densite des residus en 0.
def hall_sheather(n, tau, alpha=0.05):
    z = norm.ppf(tau)
    return n ** (-1 / 3) * norm.ppf(1 - alpha / 2) ** (2 / 3) * (1.5 * norm.pdf(z) ** 2 / (2 * z ** 2 + 1)) ** (1 / 3)


# Densite des residus en 0 (inverse de la sparsite) par estimateur a noyau.
def densite_en_zero(e, y, tau, noyau="epanechnikov"):
    n = len(e)
    h = hall_sheather(n, tau)
    iqr = np.subtract(*np.percentile(e, [75, 25]))
    h = min(np.std(y), iqr / 1.34) * (norm.ppf(tau + h) - norm.ppf(tau - h))
    return np.sum(NOYAUX[noyau](e / h)) / (n * h)


# Resultat d'une regression quantile. Les p-values suivent une loi de Student a n - k ddl.
# Le pseudo R2 de Koenker et Machado compare la perte du modele a celle du quantile constant.
class ResultatQuantile:
    def __init__(self, y, X, tau, beta, v, noms):
        self.tau, self.beta, self.cov, self.noms = tau, beta, v, noms
        self.n, self.k = X.shape
        self.resid = y - X @ beta
        self.objectif = perte(self.resid, tau)
        self.pseudo_r2 = 1 - self.objectif / perte(y - np.percentile(y, 100 * tau), tau)
        self.se = np.sqrt(np.diag(v))
        self.t = beta / self.se
        self.p = 2 * stats.t.sf(np.abs(self.t), self.n - self.k)

    def predire(self, X):
        return np.asarray(X, dtype=float) @ self.beta

    def __str__(self):
        lignes = [f"quantile tau = {self.tau}", f"{'':>12}{'coef':>12}{'se':>12}{'t':>10}{'p':>10}"]
        for i, nom in enumerate(self.noms):
            lignes.append(f"{nom:>12}{self.beta[i]:>12.4f}{self.se[i]:>12.4f}{self.t[i]:>10.2f}{self.p[i]:>10.3f}")
        lignes.append(f"n = {self.n}, pseudo R2 = {self.pseudo_r2:.4f}")
        return "\n".join(lignes)


def preparer(y, X, constante, noms):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    noms = list(noms) if noms is not None else [f"x{i}" for i in range(1, X.shape[1] + 1)]
    if constante:
        X, noms = np.column_stack([np.ones(len(X)), X]), ["const"] + noms
    return y, X, noms


# Regression quantile. vcov : "robuste" (sandwich de Powell, noyau pour la densite des residus),
# "iid" (erreurs iid) ou "bootstrap" (bootstrap par paires, B tirages). Les k residus exactement nuls
# (points d'interpolation) sont classes comme negatifs, pour que le resultat ne depende pas du bruit numerique.
def regression_quantile(y, X, tau=0.5, constante=True, vcov="robuste", noyau="epanechnikov",
                        B=200, graine=None, noms=None):
    y, X, noms = preparer(y, X, constante, noms)
    beta = resoudre(y, X, tau)
    e = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    if vcov == "bootstrap":
        rng = np.random.default_rng(graine)
        tirages = []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            tirages.append(resoudre(y[i], X[i], tau))
        v = np.cov(np.array(tirages).T)
    else:
        f0 = densite_en_zero(e, y, tau, noyau)
        if vcov == "robuste":
            d = np.where(e > 1e-9 * max(1.0, np.abs(y).max()), (tau / f0) ** 2, ((1 - tau) / f0) ** 2)
            v = bread @ (X.T * d) @ X @ bread
        elif vcov == "iid":
            v = tau * (1 - tau) / f0 ** 2 * bread
        else:
            raise ValueError("vcov inconnue : " + vcov)
    return ResultatQuantile(y, X, tau, beta, np.atleast_2d(v), noms)


# Plusieurs quantiles a la fois.
def regressions_quantiles(y, X, taus=(0.1, 0.25, 0.5, 0.75, 0.9), **kw):
    return [regression_quantile(y, X, tau, **kw) for tau in taus]


# Quantiles conditionnels predits pour de nouveaux X, un par colonne. Les droites de quantiles
# peuvent se croiser ; avec rearranger, on trie les quantiles predits pour chaque ligne
# (Chernozhukov, Fernandez-Val et Galichon), ce qui ne peut qu'ameliorer l'estimation.
def predire_quantiles(resultats, X, rearranger=True):
    P = np.column_stack([r.predire(X) for r in resultats])
    return np.sort(P, axis=1) if rearranger else P


# Test d'egalite des pentes entre quantiles (H0 : memes pentes pour tous les taus donnes).
# Bootstrap par paires pour la covariance jointe, puis test de Wald sur les differences.
# Rejeter signale une heteroscedasticite ou un effet qui varie le long de la distribution.
def egalite_pentes(y, X, taus=(0.25, 0.75), constante=True, B=200, graine=None):
    y, X, _ = preparer(y, X, constante, None)
    rng = np.random.default_rng(graine)
    k0 = int(constante)

    def pentes(yy, XX):
        return np.concatenate([resoudre(yy, XX, tau)[k0:] for tau in taus])

    obs = pentes(y, X)
    tirages = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        tirages.append(pentes(y[i], X[i]))
    p = X.shape[1] - k0
    q = len(taus) - 1
    R = np.zeros((q * p, len(taus) * p))
    for j in range(q):
        R[j * p:(j + 1) * p, j * p:(j + 1) * p] = np.eye(p)
        R[j * p:(j + 1) * p, (j + 1) * p:(j + 2) * p] = -np.eye(p)
    d = R @ obs
    stat = d @ np.linalg.solve(R @ np.cov(np.array(tirages).T) @ R.T, d)
    return stat, stats.chi2.sf(stat, q * p)
