import numpy as np
from scipy import stats

from .optimiseurs import newton, bfgs


# Jacobien numerique par differences centrees, colonne par colonne.
def jacobien(f, x, h=1e-6):
    colonnes = []
    for j in range(len(x)):
        pas = h * max(1.0, abs(x[j]))
        e = np.zeros(len(x))
        e[j] = pas
        colonnes.append((np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e))) / (2 * pas))
    return np.column_stack(colonnes)


# Hessien numerique d'une fonction scalaire, par differences centrees croisees.
def hessien_num(f, x, h=1e-4):
    p = len(x)
    pas = h * np.maximum(1.0, np.abs(x))
    H = np.zeros((p, p))
    for j in range(p):
        for k in range(j, p):
            ej, ek = np.zeros(p), np.zeros(p)
            ej[j], ek[k] = pas[j], pas[k]
            H[j, k] = H[k, j] = (
                f(x + ej + ek) - f(x + ej - ek) - f(x - ej + ek) + f(x - ej - ek)
            ) / (4 * pas[j] * pas[k])
    return H


# Resultat d'un maximum de vraisemblance. Les p-values suivent la loi normale.
class ResultatMLE:
    def __init__(self, theta, v, noms, loglik, ll_nul, n, converge, iterations):
        self.theta = theta
        self.beta = theta
        self.cov = v
        self.noms = noms
        self.loglik = loglik
        self.n = n
        self.k = len(theta)
        self.converge = converge
        self.iterations = iterations
        self.se = np.sqrt(np.diag(v))
        self.z = theta / self.se
        self.p = 2 * stats.norm.sf(np.abs(self.z))
        self.aic = 2 * self.k - 2 * loglik
        self.bic = self.k * np.log(n) - 2 * loglik
        self.pseudo_r2 = np.nan if ll_nul is None else 1 - loglik / ll_nul

    def __str__(self):
        lignes = [f"{'':>16}{'coef':>12}{'se':>12}{'z':>10}{'p':>10}"]
        for i, nom in enumerate(self.noms):
            lignes.append(
                f"{nom:>16}{self.theta[i]:>12.4f}{self.se[i]:>12.4f}"
                f"{self.z[i]:>10.2f}{self.p[i]:>10.3f}"
            )
        lignes.append(f"n = {self.n}, log-vraisemblance = {self.loglik:.3f}, AIC = {self.aic:.1f}, BIC = {self.bic:.1f}")
        if not np.isnan(self.pseudo_r2):
            lignes.append(f"pseudo R2 de McFadden = {self.pseudo_r2:.4f}")
        return "\n".join(lignes)


# Maximise la somme de ll_obs(theta). Si scores et hessien analytiques sont donnes
# on les utilise, sinon on derive numeriquement. Avec "bfgs", on termine par
# Newton pour avoir un optimum precis. transformation passe des parametres
# d'estimation aux parametres affiches, la variance suit par delta method.
def estimer(ll_obs, theta0, noms, cov="hessien", groupes=None, methode="bfgs",
            scores=None, hessien=None, transformation=None, ll_nul=None):
    theta0 = np.asarray(theta0, dtype=float)
    f = lambda t: -np.sum(ll_obs(t))
    if scores is None:
        scores = lambda t: jacobien(ll_obs, t)
    grad = lambda t: -scores(t).sum(axis=0)
    if hessien is None:
        hess = lambda t: hessien_num(f, t)
    else:
        hess = lambda t: -hessien(t)

    theta = theta0
    with np.errstate(all="ignore"):
        if methode == "bfgs":
            theta, _, _ = bfgs(f, grad, theta)
        theta, it, ok = newton(f, grad, hess, theta)

    S = scores(theta)
    A_inv = np.linalg.inv(hess(theta))
    if cov == "hessien":
        v = A_inv
    elif cov == "opg":
        v = np.linalg.inv(S.T @ S)
    elif cov == "robuste":
        v = A_inv @ (S.T @ S) @ A_inv
    elif cov == "cluster":
        codes = np.unique(groupes, return_inverse=True)[1]
        g = np.zeros((codes.max() + 1, S.shape[1]))
        np.add.at(g, codes, S)
        v = A_inv @ (g.T @ g) @ A_inv
    else:
        raise ValueError("cov inconnue : " + cov)

    loglik = np.sum(ll_obs(theta))
    n = len(ll_obs(theta))
    if transformation is not None:
        J = jacobien(transformation, theta)
        v = J @ v @ J.T
        theta = transformation(theta)
    return ResultatMLE(theta, v, noms, loglik, ll_nul, n, ok, it)


# Test de Wald de H0 : R theta = q. Renvoie (statistique, p-value), loi du chi2.
def wald(res, R, q=None):
    R = np.atleast_2d(R)
    q = np.zeros(R.shape[0]) if q is None else np.asarray(q, dtype=float)
    d = R @ res.theta - q
    stat = d @ np.linalg.solve(R @ res.cov @ R.T, d)
    return stat, stats.chi2.sf(stat, R.shape[0])


# Test du rapport de vraisemblance entre un modele restreint et le modele complet.
def rapport_vraisemblance(restreint, complet, ddl=None):
    ddl = complet.k - restreint.k if ddl is None else ddl
    stat = 2 * (complet.loglik - restreint.loglik)
    return stat, stats.chi2.sf(stat, ddl)
