import math

import numpy as np
from scipy.linalg import solve_triangular

from .rng import Rng


# Version Python de reference des deux echantillonneurs. Elle suit exactement les memes etapes
# et le meme ordre de tirages que les cœurs C++ et Rust, ce qui permet de comparer les chaines
# tirage par tirage. Elle est lente : elle sert de specification et de test.
def rng_flux(graine, n):
    r = Rng(graine)
    return np.array([r.uniforme() for _ in range(n)] + [r.normale() for _ in range(n)]
                    + [r.gamma(2.5) for _ in range(n)] + [r.gamma(0.4) for _ in range(n)])


# Un balayage de Gibbs : effets individuels, coefficients, variance des erreurs, variance des effets.
# etat est un dictionnaire (beta, alpha, sigma2, tau2) modifie sur place.
def balayage(rng, y, X, grp, ng, etat, hyp, XtX, Tg):
    var_beta, a_s, b_s, a_t, b_t = hyp
    n, k = X.shape
    sigma2, tau2 = etat["sigma2"], etat["tau2"]

    S = np.bincount(grp, weights=y - X @ etat["beta"], minlength=ng)
    prec = Tg / sigma2 + 1.0 / tau2
    moy = (S / sigma2) / prec
    z = np.array([rng.normale() for _ in range(ng)])
    alpha = moy + z / np.sqrt(prec)

    rhs = X.T @ (y - alpha[grp]) / sigma2
    P = XtX / sigma2 + np.eye(k) / var_beta
    L = np.linalg.cholesky(P)
    m = solve_triangular(L.T, solve_triangular(L, rhs, lower=True), lower=False)
    zb = np.array([rng.normale() for _ in range(k)])
    beta = m + solve_triangular(L.T, zb, lower=False)

    r = y - X @ beta - alpha[grp]
    sigma2 = (b_s + 0.5 * float(r @ r)) / rng.gamma(a_s + 0.5 * n)
    tau2 = (b_t + 0.5 * float(alpha @ alpha)) / rng.gamma(a_t + 0.5 * ng)
    etat.update(beta=beta, alpha=alpha, sigma2=sigma2, tau2=tau2)


def gibbs_panel(y, X, grp, ng, hyp, init, n_iter, burn, thin, graine):
    rng = Rng(graine)
    k = X.shape[1]
    XtX = X.T @ X
    Tg = np.bincount(grp, minlength=ng).astype(float)
    etat = dict(beta=init[:k].copy(), alpha=init[k:k + ng].copy(), sigma2=float(init[k + ng]),
                tau2=float(init[k + ng + 1]))
    sorties, somme = [], np.zeros(ng)
    for it in range(n_iter):
        balayage(rng, y, X, grp, ng, etat, hyp, XtX, Tg)
        if it >= burn and (it - burn) % thin == 0:
            sorties.append(np.concatenate([etat["beta"], [etat["sigma2"], etat["tau2"]]]))
            somme += etat["alpha"]
    return np.array(sorties), somme / len(sorties)


# Log de la loi a posteriori du logit (a une constante pres), prior N(0, sd^2 I).
def log_posterieure(y, X, sd, b):
    eta = X @ b
    sp = np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
    return float(np.sum(y * eta - sp)) - 0.5 * float(b @ b) / (sd * sd)


def mh_logit(y, X, sd, L, x0, n_iter, burn, thin, graine):
    rng = Rng(graine)
    k = X.shape[1]
    b = x0.copy()
    lp = log_posterieure(y, X, sd, b)
    sorties, acc = [], 0
    for it in range(n_iter):
        z = np.array([rng.normale() for _ in range(k)])
        prop = b + L @ z
        lp2 = log_posterieure(y, X, sd, prop)
        u = rng.uniforme()
        if math.log(u) < lp2 - lp:
            b, lp = prop, lp2
            acc += 1
        if it >= burn and (it - burn) % thin == 0:
            sorties.append(b.copy())
    return np.array(sorties), acc
