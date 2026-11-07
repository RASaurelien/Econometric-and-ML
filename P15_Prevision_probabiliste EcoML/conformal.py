import numpy as np
from scipy.stats import norm


def conformal_quantile(scores, level):
    """Quantile empirique avec la correction de taille finie : le ceil((n + 1) * level)-ieme
    plus petit score. Renvoie +inf si le niveau est trop haut pour n et -inf s'il est <= 0."""
    n = len(scores)
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return np.inf
    if k < 1:
        return -np.inf
    return np.sort(scores)[k - 1]


def nonconformity(y, lo_base, hi_base):
    """Score de la regression quantile conforme. Avec lo_base = hi_base = prevision ponctuelle,
    c'est le residu absolu."""
    return np.maximum(lo_base - y, y - hi_base)


def gaussian_interval(pred, calib_resid, alpha):
    sigma = np.std(calib_resid)
    z = norm.ppf(1 - alpha / 2)
    return pred - z * sigma, pred + z * sigma


def split_conformal(lo_base, hi_base, calib_scores, alpha):
    """Intervalle conforme classique : une seule calibration, valable si les donnees
    sont echangeables."""
    q = conformal_quantile(calib_scores, 1 - alpha)
    return lo_base - q, hi_base + q


def online_conformal(y, lo_base, hi_base, calib_scores, alpha, window=200, gamma=0.0):
    """Conformal en ligne : a chaque date, le quantile est calcule sur les `window` derniers
    scores deja observes. Avec gamma > 0, le niveau alpha_t est ajuste apres chaque
    observation (Adaptive Conformal Inference, Gibbs et Candes, 2021). gamma = 0 donne le
    conformal glissant sans adaptation."""
    scores = list(calib_scores)
    a = alpha
    T = len(y)
    lo, hi, alphas = np.empty(T), np.empty(T), np.empty(T)
    for t in range(T):
        recent = np.asarray(scores[-window:])
        Q = conformal_quantile(recent, 1 - a)
        if Q == -np.inf:
            lo[t], hi[t] = np.inf, -np.inf
        else:
            lo[t], hi[t] = lo_base[t] - Q, hi_base[t] + Q
        err = float(not (lo[t] <= y[t] <= hi[t]))
        alphas[t] = a
        a = a + gamma * (alpha - err)
        scores.append(nonconformity(y[t], lo_base[t], hi_base[t]))
    return lo, hi, alphas
