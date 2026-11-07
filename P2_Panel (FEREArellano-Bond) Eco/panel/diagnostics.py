import numpy as np
from scipy import stats


# Test de Wald de H0 : R beta = q. Renvoie (statistique, p-value), loi du chi2.
def wald(res, R, q=None):
    R = np.atleast_2d(R)
    q = np.zeros(R.shape[0]) if q is None else np.asarray(q, dtype=float)
    d = R @ res.beta - q
    stat = d @ np.linalg.solve(R @ res.cov @ R.T, d)
    return stat, stats.chi2.sf(stat, R.shape[0])


# Hausman : effets fixes contre effets aleatoires, sur les pentes uniquement.
# H0 : les effets individuels ne sont pas correles aux regresseurs.
def hausman(fe, re):
    d = fe.beta - re.beta[1:]
    V = fe.cov - re.cov[1:, 1:]
    stat = d @ np.linalg.solve(V, d)
    return stat, stats.chi2.sf(stat, len(d))


# Test F de presence d'effets individuels, MCO empiles contre effets fixes.
def f_effets_fixes(empile, fe):
    ddl1 = fe.N - 1
    stat = ((empile.rss - fe.rss) / ddl1) / (fe.rss / fe.ddl)
    return stat, stats.f.sf(stat, ddl1, fe.ddl)


# Version robuste du Hausman : on teste que les coefficients des moyennes
# individuelles sont nuls dans la regression de Mundlak.
def test_mundlak(res):
    K = len(res.idx_moyennes)
    R = np.zeros((K, res.k))
    R[np.arange(K), res.idx_moyennes] = 1
    return wald(res, R)
