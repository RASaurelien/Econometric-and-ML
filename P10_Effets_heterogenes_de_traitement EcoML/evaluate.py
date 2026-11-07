import numpy as np
from scipy.stats import spearmanr


def pehe(tau_true, tau_hat):
    return np.sqrt(np.mean((tau_true - tau_hat) ** 2))


def qini_curve(tau_hat, T, Y):
    # test randomise : on classe par effet predit decroissant
    order = np.argsort(-tau_hat)
    t, y = T[order], Y[order]
    cum_t, cum_c = np.cumsum(t), np.cumsum(1 - t)
    sum_t, sum_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    gain = sum_t - sum_c * np.where(cum_c > 0, cum_t / np.maximum(cum_c, 1), 0)
    frac = np.arange(1, len(y) + 1) / len(y)
    return frac, gain


def auuc(tau_hat, T, Y):
    frac, gain = qini_curve(tau_hat, T, Y)
    random = gain[-1] * frac
    diff = gain - random
    return np.sum((diff[1:] + diff[:-1]) / 2 * np.diff(frac))


def group_effects(tau_hat, tau_true, n_groups=5):
    cuts = np.quantile(tau_hat, np.linspace(0, 1, n_groups + 1))
    groups = np.clip(np.searchsorted(cuts, tau_hat, side="right") - 1, 0, n_groups - 1)
    return np.array([[tau_hat[groups == g].mean(), tau_true[groups == g].mean()]
                     for g in range(n_groups)])


def summary(tau_true, tau_hat, T, Y):
    return {
        "PEHE": pehe(tau_true, tau_hat),
        "corr rangs": spearmanr(tau_true, tau_hat)[0],
        "AUUC": auuc(tau_hat, T, Y),
    }
