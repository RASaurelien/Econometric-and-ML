import numpy as np
import pandas as pd
import statsmodels.api as sm


def pooled_ols(df):
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    a, b = np.linalg.lstsq(X, df["y"].values, rcond=None)[0]
    return b, lambda d: a + b * d["x"].values


def _demean_two_way(v, unit, time, cu, ct, n_units, n_times, tol=1e-10, max_iter=5000):
    v = v.astype(float).copy()
    for _ in range(max_iter):
        before = v.copy()
        v -= (np.bincount(unit, v, n_units) / cu)[unit]
        v -= (np.bincount(time, v, n_times) / ct)[time]
        if np.max(np.abs(v - before)) < tol:
            break
    return v


def twfe(df, n_units, n_times, tol=1e-10, max_iter=5000):
    """Effets fixes individuels et temporels, panel eventuellement desequilibre.
    Les moyennes sont retirees par projections alternees (equivalent a la regression avec dummies)."""
    unit, time = df["unit"].values, df["time"].values
    cu = np.maximum(np.bincount(unit, minlength=n_units), 1)
    ct = np.maximum(np.bincount(time, minlength=n_times), 1)
    x_t = _demean_two_way(df["x"].values, unit, time, cu, ct, n_units, n_times, tol, max_iter)
    y_t = _demean_two_way(df["y"].values, unit, time, cu, ct, n_units, n_times, tol, max_iter)
    beta = (x_t @ y_t) / (x_t @ x_t)

    r = df["y"].values - beta * df["x"].values
    mu = r.mean()
    a, g = np.zeros(n_units), np.zeros(n_times)
    for _ in range(max_iter):
        a_new = np.bincount(unit, r - mu - g[time], n_units) / cu
        g_new = np.bincount(time, r - mu - a_new[unit], n_times) / ct
        done = max(np.abs(a_new - a).max(), np.abs(g_new - g).max()) < tol
        a, g = a_new, g_new
        if done:
            break
    return beta, lambda d: beta * d["x"].values + mu + a[d["unit"].values] + g[d["time"].values]


def interactive_fe(df, n_units, n_times, rank=2, n_iter=3000, tol=1e-9):
    """Effets fixes interactifs de Bai (2009) par moindres carres alternes : beta par MCO sur
    y moins la partie factorielle, puis effets additifs et facteurs par SVD des residus.
    Les cellules non observees sont imputees avec l'ajustement courant."""
    Y = np.zeros((n_units, n_times))
    X = np.zeros((n_units, n_times))
    M = np.zeros((n_units, n_times), dtype=bool)
    u, t = df["unit"].values, df["time"].values
    Y[u, t], X[u, t], M[u, t] = df["y"].values, df["x"].values, True

    beta = (X[M] @ Y[M]) / (X[M] @ X[M])
    fit = np.zeros_like(Y)
    for _ in range(n_iter):
        R = np.where(M, Y - beta * X, fit)
        mu = R.mean()
        a = R.mean(axis=1, keepdims=True) - mu
        g = R.mean(axis=0, keepdims=True) - mu
        resid = R - mu - a - g
        if rank > 0:
            U, s, Vt = np.linalg.svd(resid, full_matrices=False)
            low = (U[:, :rank] * s[:rank]) @ Vt[:rank]
        else:
            low = np.zeros_like(resid)
        new_fit = mu + a + g + low
        new_beta = (X[M] @ (Y - new_fit)[M]) / (X[M] @ X[M])
        done = abs(new_beta - beta) < tol
        beta, fit = new_beta, new_fit
        if done:
            break
    return beta, lambda d: beta * d["x"].values + fit[d["unit"].values, d["time"].values]


def pooled_cluster_se(df):
    fit = sm.OLS(df["y"].values, sm.add_constant(df["x"].values)).fit(
        cov_type="cluster", cov_kwds={"groups": df["unit"].values})
    return fit.params[1], fit.bse[1]


def twfe_cluster_se(df):
    """Meme estimateur que `twfe` mais avec dummies, pour obtenir des ecarts-types clusterises par unite."""
    D_u = pd.get_dummies(df["unit"], prefix="u", drop_first=True).astype(float)
    D_t = pd.get_dummies(df["time"], prefix="t", drop_first=True).astype(float)
    X = np.column_stack([df["x"].values, D_u.values, D_t.values])
    fit = sm.OLS(df["y"].values, sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": df["unit"].values})
    return fit.params[1], fit.bse[1]


def add_group_means(train, test, n_units, n_times):
    """Moyennes de x par unite et par periode, calculees sur l'echantillon d'entrainement
    (variables de Mundlak). Les groupes absents de l'entrainement recoivent la moyenne globale."""
    tr, te = train.copy(), test.copy()
    glob = tr["x"].mean()
    cnt_u = np.bincount(tr["unit"].values, minlength=n_units)
    cnt_t = np.bincount(tr["time"].values, minlength=n_times)
    mean_u = np.where(cnt_u > 0, np.bincount(tr["unit"].values, tr["x"].values, n_units) / np.maximum(cnt_u, 1), glob)
    mean_t = np.where(cnt_t > 0, np.bincount(tr["time"].values, tr["x"].values, n_times) / np.maximum(cnt_t, 1), glob)
    for d in (tr, te):
        d["xbar_u"] = mean_u[d["unit"].values]
        d["xbar_t"] = mean_t[d["time"].values]
    return tr, te


def ridge_effects(train, n_units, n_times, mundlak=False, lambdas=(0.1, 0.3, 1.0, 3.0, 10.0),
                  val_share=0.1, seed=0):
    """Effets individuels et temporels penalises en L2 (effets aleatoires). C'est la version
    lineaire exacte de ce qu'apprend un embedding additif avec penalite. Avec mundlak=True, on
    ajoute les moyennes de x par unite et par periode pour corriger la correlation entre x et les effets."""
    n_free = 1 + (3 if mundlak else 1)

    def design(df):
        cols = [df["x"].values] + ([df["xbar_u"].values, df["xbar_t"].values] if mundlak else [])
        U = np.zeros((len(df), n_units))
        U[np.arange(len(df)), df["unit"].values] = 1.0
        Tm = np.zeros((len(df), n_times))
        Tm[np.arange(len(df)), df["time"].values] = 1.0
        return np.column_stack([np.ones(len(df)), np.column_stack(cols), U, Tm])

    def solve(X, y, lam):
        pen = np.r_[np.zeros(n_free), np.full(n_units + n_times, lam)]
        return np.linalg.solve(X.T @ X + np.diag(pen) + 1e-9 * np.eye(X.shape[1]), X.T @ y)

    X, y = design(train), train["y"].values
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_val = max(int(val_share * len(y)), 1)
    va, tr = perm[:n_val], perm[n_val:]
    val_err = [np.mean((y[va] - X[va] @ solve(X[tr], y[tr], lam)) ** 2) for lam in lambdas]
    lam = lambdas[int(np.argmin(val_err))]
    coef = solve(X, y, lam)
    return coef[1], (lambda d: design(d) @ coef), lam
