import numpy as np
from scipy.special import expit

from . import coeurs
from .diagnostics import Resume


# Resultat d'un echantillonnage : tirages (chaines, n_garde, parametres), noms et quelques extras.
class ResultatBayes:
    def __init__(self, tirages, noms, backend, **extras):
        self.tirages, self.noms, self.backend = tirages, list(noms), backend
        self.__dict__.update(extras)

    def moyennes(self):
        return self.tirages.mean(axis=(0, 1))

    def resume(self, niveau=0.95):
        return Resume(self.tirages, self.noms, niveau)


def preparer(y, X, constante, noms):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    noms = list(noms) if noms is not None else [f"x{i}" for i in range(1, X.shape[1] + 1)]
    if constante:
        X, noms = np.column_stack([np.ones(len(X)), X]), ["const"] + noms
    return y, X, noms


# Panel lineaire a effets aleatoires, par Gibbs :
#   y_it = x_it'b + a_i + e_it,  a_i ~ N(0, tau2),  e_it ~ N(0, sigma2).
# Priors : b ~ N(0, prior_var_beta I), sigma2 ~ IG(a_sigma, b_sigma), tau2 ~ IG(a_tau, b_tau).
# Les chaines partent de points disperses autour des MCO. backend : "rust", "cpp", "python" ou "auto".
# alpha_moyen contient la moyenne a posteriori de chaque effet individuel.
def gibbs_panel(y, X, ids, n_iter=3000, burn=1000, thin=1, chaines=4, graine=0, backend="auto",
                prior_var_beta=1e4, a_sigma=0.01, b_sigma=0.01, a_tau=1.0, b_tau=1.0,
                constante=True, noms=None):
    y, X, noms = preparer(y, X, constante, noms)
    backend = coeurs.choisir(backend)
    groupes, grp = np.unique(ids, return_inverse=True)
    ng, k = len(groupes), X.shape[1]
    hyp = [prior_var_beta, a_sigma, b_sigma, a_tau, b_tau]
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = np.sum((y - X @ beta0) ** 2) / (len(y) - k)
    se0 = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    tirages, alphas = [], []
    for c in range(chaines):
        r = np.random.default_rng(graine + c)
        init = np.concatenate([beta0 + 3 * se0 * r.normal(size=k), np.zeros(ng),
                               [s2 * np.exp(0.5 * r.normal()), np.exp(r.normal())]])
        t, a = coeurs.gibbs(backend, y, X, grp, ng, init, hyp, n_iter, burn, thin, graine + c)
        tirages.append(t)
        alphas.append(a)
    return ResultatBayes(np.array(tirages), noms + ["sigma2", "tau2"], backend,
                         alpha_moyen=np.mean(alphas, axis=0), groupes=groupes)


# Mode a posteriori et hessien d'un logit avec prior N(0, sd^2 I), par Newton-Raphson.
def mode_logit(y, X, sd):
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = expit(X @ b)
        H = (X * (p * (1 - p))[:, None]).T @ X + np.eye(len(b)) / sd ** 2
        pas = np.linalg.solve(H, X.T @ (y - p) - b / sd ** 2)
        b = b + pas
        if np.max(np.abs(pas)) < 1e-10:
            break
    return b, H


# Logit bayesien par Metropolis-Hastings a marche aleatoire. La proposition suit la loi normale
# d'approximation de Laplace (covariance (H)^-1) multipliee par echelle / sqrt(k), avec echelle = 2.38
# (regle de Roberts, Gelman et Gilks). Les chaines partent de points disperses autour du mode.
# acceptation donne le taux d'acceptation de chaque chaine.
def mh_logit(y, X, n_iter=6000, burn=2000, thin=1, chaines=4, graine=0, backend="auto",
             prior_sd=10.0, echelle=2.38, constante=True, noms=None):
    y, X, noms = preparer(y, X, constante, noms)
    backend = coeurs.choisir(backend)
    k = X.shape[1]
    mode, H = mode_logit(y, X, prior_sd)
    L0 = np.linalg.cholesky(np.linalg.inv(H))
    L = L0 * echelle / np.sqrt(k)
    tirages, taux = [], []
    for c in range(chaines):
        x0 = mode + 2 * L0 @ np.random.default_rng(graine + c).normal(size=k)
        t, acc = coeurs.mh(backend, y, X, prior_sd, L, x0, n_iter, burn, thin, graine + c)
        tirages.append(t)
        taux.append(acc / n_iter)
    return ResultatBayes(np.array(tirages), noms, backend, acceptation=taux, mode=mode)
