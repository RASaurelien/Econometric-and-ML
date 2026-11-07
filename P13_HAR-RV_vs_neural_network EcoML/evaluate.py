import numpy as np
from scipy import stats


def qlike(rv, forecast):
    ratio = rv / forecast
    return ratio - np.log(ratio) - 1


def squared_error(rv, forecast):
    return (rv - forecast) ** 2


def diebold_mariano(loss1, loss2, h=1):
    """Test de Diebold-Mariano sur deux series de pertes, avec la correction
    de Harvey, Leybourne et Newbold. H0 : meme precision de prevision."""
    d = np.asarray(loss1) - np.asarray(loss2)
    T = len(d)
    dbar = d.mean()
    var = np.mean((d - dbar) ** 2)
    for k in range(1, h):
        var += 2 * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    stat = dbar / np.sqrt(max(var, 1e-18) / T)
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    pval = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return stat, pval


def mincer_zarnowitz(realized, forecast):
    """Regression de la realisation sur la prevision : constante, pente et R2."""
    X = np.column_stack([np.ones(len(forecast)), forecast])
    coef = np.linalg.lstsq(X, realized, rcond=None)[0]
    resid = realized - X @ coef
    r2 = 1 - resid @ resid / np.sum((realized - realized.mean()) ** 2)
    return coef[0], coef[1], r2
