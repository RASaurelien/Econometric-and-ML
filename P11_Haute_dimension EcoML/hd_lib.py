import numpy as np
import statsmodels.api as sm
from scipy.linalg import toeplitz
from scipy.stats import norm
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV, RidgeCV


def simulate(n, p, s, rho, alpha, rng, confound=0.8):
    cov = toeplitz(rho ** np.arange(p))
    X = rng.normal(size=(n, p)) @ np.linalg.cholesky(cov).T
    beta = np.zeros(p)
    beta[:s] = np.array([(-1) ** j / (1 + 0.1 * j) for j in range(s)]) * 2
    gamma = np.zeros(p)
    gamma[:s] = confound / (1 + 0.1 * np.arange(s))
    D = X @ gamma + rng.normal(size=n)
    Y = alpha * D + X @ beta + rng.normal(size=n)
    return Y, D, X, beta


def plugin_lasso(X, y, c=1.1, gamma=0.05, iters=3):
    # penalite de Belloni et al., adaptee a la parametrisation de sklearn
    n, p = X.shape
    sigma = y.std()
    for _ in range(iters):
        pen = c * sigma * norm.ppf(1 - gamma / (2 * p)) / np.sqrt(n)
        model = Lasso(alpha=pen, max_iter=5000).fit(X, y)
        resid = y - model.predict(X)
        s_hat = max(np.count_nonzero(model.coef_), 1)
        sigma = np.sqrt(resid @ resid / max(n - s_hat, 1))
    return model


def fit_selector(name, X, y):
    if name == "lasso_cv":
        return LassoCV(cv=5, max_iter=5000).fit(X, y)
    if name == "lasso_plugin":
        return plugin_lasso(X, y)
    if name == "elastic_net":
        return ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=5, max_iter=5000).fit(X, y)
    if name == "ridge":
        return RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(X, y)
    raise ValueError(name)


def support_scores(coef_hat, beta, tol=1e-8):
    chosen = np.abs(coef_hat) > tol
    truth = beta != 0
    tp = np.sum(chosen & truth)
    precision = tp / chosen.sum() if chosen.sum() else 0.0
    recall = tp / truth.sum()
    return precision, recall


def _ols_hc1(Y, Z):
    fit = sm.OLS(Y, sm.add_constant(Z)).fit(cov_type="HC1")
    return fit.params[1], fit.bse[1]


def naive_ols(Y, D, X):
    return _ols_hc1(Y, D)


def naive_post_lasso(Y, D, X, selector="lasso_cv"):
    # selection de X dans l'equation de Y seulement, D est ensuite ajoutee
    keep = np.flatnonzero(fit_selector(selector, X, Y).coef_)
    Z = np.column_stack([D, X[:, keep]]) if len(keep) else D
    return _ols_hc1(Y, Z)


def post_double_selection(Y, D, X, selector="lasso_plugin"):
    sel_y = np.flatnonzero(fit_selector(selector, X, Y).coef_)
    sel_d = np.flatnonzero(fit_selector(selector, X, D).coef_)
    keep = np.union1d(sel_y, sel_d).astype(int)
    Z = np.column_stack([D, X[:, keep]]) if len(keep) else D
    return _ols_hc1(Y, Z)
