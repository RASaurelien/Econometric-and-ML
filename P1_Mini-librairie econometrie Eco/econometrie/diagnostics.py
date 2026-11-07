import numpy as np
from scipy import stats


# Test de Wald de H0 : R beta = q. Renvoie (statistique, p-value), loi du chi2.
def wald(res, R, q=None):
    R = np.atleast_2d(R)
    q = np.zeros(R.shape[0]) if q is None else np.asarray(q, dtype=float)
    d = R @ res.beta - q
    stat = d @ np.linalg.solve(R @ res.cov @ R.T, d)
    return stat, stats.chi2.sf(stat, R.shape[0])


# Breusch-Pagan version Koenker : n * R2 de la regression de e^2 sur X.
# Il faut une constante dans X.
def breusch_pagan(res):
    e2 = res.residus ** 2
    g = np.linalg.lstsq(res.X, e2, rcond=None)[0]
    ecart = e2 - res.X @ g
    r2 = 1 - ecart @ ecart / ((e2 - e2.mean()) @ (e2 - e2.mean()))
    stat = res.n * r2
    return stat, stats.chi2.sf(stat, res.k - 1)


# Hausman : compare les coefficients IV et MCO, sous H0 de regresseurs exogenes.
# indices permet de ne tester que certains coefficients (les endogenes en general).
def hausman(res_iv, res_mco, indices=None):
    idx = list(range(res_iv.k)) if indices is None else list(indices)
    d = res_iv.beta[idx] - res_mco.beta[idx]
    V = res_iv.cov[np.ix_(idx, idx)] - res_mco.cov[np.ix_(idx, idx)]
    stat = d @ np.linalg.solve(V, d)
    return stat, stats.chi2.sf(stat, len(idx))


# J de Hansen : teste la validite des instruments en exces.
# Possible seulement si le modele est sur-identifie.
def j_hansen(res):
    ddl = res.Z.shape[1] - res.k
    if ddl <= 0:
        raise ValueError("modele juste identifie, pas de test de sur-identification")
    g = res.Z.T @ res.residus / res.n
    S = (res.Z * (res.residus ** 2)[:, None]).T @ res.Z / res.n
    stat = res.n * g @ np.linalg.solve(S, g)
    return stat, stats.chi2.sf(stat, ddl)
