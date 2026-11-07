import numpy as np


def interval_metrics(y, lo, hi, alpha, regime, big):
    """big : valeur qui remplace les bornes infinies pour calculer largeurs et scores."""
    empty = hi < lo
    infinite = np.isinf(lo) | np.isinf(hi)
    lo_f = np.where(np.isinf(lo) & ~empty, -big, lo)
    hi_f = np.where(np.isinf(hi) & ~empty, big, hi)
    covered = (lo <= y) & (y <= hi)
    width = np.where(empty, 0.0, hi_f - lo_f)
    score = width + (2 / alpha) * np.maximum(lo_f - y, 0) + (2 / alpha) * np.maximum(y - hi_f, 0)
    score = np.where(empty, (2 / alpha) * np.abs(y), score)
    return {
        "couverture": covered.mean(),
        "couv. regime calme": covered[regime == 0].mean() if (regime == 0).any() else np.nan,
        "couv. regime agite": covered[regime == 1].mean() if (regime == 1).any() else np.nan,
        "largeur moyenne": np.minimum(width, big).mean(),
        "score d'intervalle": score.mean(),
        "part infinie": infinite.mean(),
    }


def rolling_coverage(y, lo, hi, window=200):
    covered = ((lo <= y) & (y <= hi)).astype(float)
    kernel = np.ones(window) / window
    return np.convolve(covered, kernel, mode="valid")


def worst_window_gap(y, lo, hi, alpha, window=200):
    r = rolling_coverage(y, lo, hi, window)
    return np.max(np.abs(r - (1 - alpha)))
