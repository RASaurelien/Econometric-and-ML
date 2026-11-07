import numpy as np

from .estimateurs import MCOBatch, poids_hc


# Wild bootstrap avec hypothese nulle imposee (beta_j = b0). X (C, n, k), y (C, n),
# w (C, B, n) sont les poids du bootstrap, par exemple +1 ou -1 avec proba 1/2.
# Renvoie la statistique observee (C,) et les statistiques bootstrap (C, B).
def wild_t(X, y, j, b0, w, variante="HC1"):
    obs = MCOBatch(y, X)
    t_obs = (obs.beta[:, j] - b0) / obs.se(variante)[:, j]
    xj = X[:, :, j]
    Xr = np.delete(X, j, axis=2)
    yr = y - b0 * xj
    e_r = MCOBatch(yr, Xr).e if Xr.shape[2] else yr
    y_star = (yr - e_r + b0 * xj)[:, None, :] + e_r[:, None, :] * w
    beta_s = np.einsum("ckn,cbn->cbk", obs.A, y_star)
    e_s = y_star - np.einsum("cnk,cbk->cbn", X, beta_s)
    if variante == "classique":
        var = obs.inv[:, j, j][:, None] * (e_s ** 2).sum(-1) / (obs.n - obs.k)
    else:
        om = poids_hc(variante, obs.h, obs.n, obs.k)
        var = np.einsum("cn,cbn->cb", obs.A[:, j, :] ** 2 * om, e_s ** 2)
    return t_obs, (beta_s[:, :, j] - b0) / np.sqrt(var)


# Bootstrap par paires (on retire des observations entieres), percentile-t.
# idx (C, B, n) contient les indices tires. Les t* sont centres sur l'estimation observee.
def pairs_t(X, y, j, b0, idx, variante="HC1"):
    obs = MCOBatch(y, X)
    t_obs = (obs.beta[:, j] - b0) / obs.se(variante)[:, j]
    Xs = np.take_along_axis(X[:, None], idx[..., None], axis=2)
    ys = np.take_along_axis(y[:, None], idx, axis=2)
    bs = MCOBatch(ys, Xs)
    return t_obs, (bs.beta[..., j] - obs.beta[:, None, j]) / bs.se(variante)[..., j]


# Wild : p-value symetrique, on rejette si elle est sous le niveau.
def rejet_wild(t_obs, t_star, niveau):
    return (np.abs(t_star) >= np.abs(t_obs)[:, None]).mean(axis=1) < niveau


# Paires : on rejette si |t| depasse le quantile de |t*|.
def rejet_pairs(t_obs, t_star, niveau):
    return np.abs(t_obs) > np.quantile(np.abs(t_star), 1 - niveau, axis=1)


# Test bootstrap sur de nombreuses replications, par blocs pour limiter la memoire.
# Renvoie un tableau de booleens (R,) : le test rejette-t-il H0 : beta_j = b0 ?
def rejets(X, y, j, b0, methode="wild", B=199, niveau=0.05, variante="HC1", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    R, n, k = X.shape
    bloc = max(1, 3_000_000 // (B * n * k))
    sortie = np.empty(R, dtype=bool)
    for debut in range(0, R, bloc):
        s = slice(debut, debut + bloc)
        C = len(y[s])
        if methode == "wild":
            w = rng.choice([-1.0, 1.0], size=(C, B, n))
            t_obs, t_star = wild_t(X[s], y[s], j, b0, w, variante)
            sortie[s] = rejet_wild(t_obs, t_star, niveau)
        else:
            idx = rng.integers(0, n, size=(C, B, n))
            t_obs, t_star = pairs_t(X[s], y[s], j, b0, idx, variante)
            sortie[s] = rejet_pairs(t_obs, t_star, niveau)
    return sortie
