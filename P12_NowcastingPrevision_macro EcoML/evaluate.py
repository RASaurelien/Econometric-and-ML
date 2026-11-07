import numpy as np
from scipy import stats


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def diebold_mariano(e1, e2, h=1):
    """Test de Diebold-Mariano (perte quadratique) avec la correction de
    Harvey, Leybourne et Newbold. H0 : meme precision de prevision."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    T = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        var += 2 * gk
    dm = dbar / np.sqrt(var / T)
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * corr
    pval = 2 * (1 - stats.t.cdf(abs(dm_hln), df=T - 1))
    return dm_hln, pval
