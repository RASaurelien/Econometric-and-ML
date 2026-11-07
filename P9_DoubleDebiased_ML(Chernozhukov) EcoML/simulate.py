import numpy as np
from scipy.linalg import toeplitz
from scipy.special import expit


def draw_x(n, p, rng, rho=0.7):
    chol = np.linalg.cholesky(toeplitz(rho ** np.arange(p)))
    return rng.normal(size=(n, p)) @ chol.T


def plr_nuisance(X, design):
    """Renvoie m(X) = E[D|X] et g(X), la partie de Y qui ne passe pas par D."""
    if design == "linear":
        m = 0.8 * X[:, 0] + 0.4 * X[:, 1] - 0.3 * X[:, 2]
        g = X[:, 0] + 0.5 * X[:, 2] - 0.4 * X[:, 3]
    elif design == "smooth":
        m = X[:, 0] + 0.25 * expit(X[:, 2])
        g = expit(X[:, 0]) + 0.25 * X[:, 2]
    else:
        m = X[:, 0] * X[:, 1] + 0.5 * np.sign(X[:, 2]) + 0.3 * np.maximum(X[:, 3], 0)
        g = np.sin(X[:, 0]) + (X[:, 1] > 0) * X[:, 2] + 0.5 * np.abs(X[:, 3])
    return m, g


def dgp_plr(n, p, theta, rng, design="smooth"):
    X = draw_x(n, p, rng)
    m, g = plr_nuisance(X, design)
    D = m + rng.normal(size=n)
    Y = theta * D + g + rng.normal(size=n)
    return Y, D, X


def irm_nuisance(X, theta, design):
    """Renvoie le score de propension, E[Y|X, D=0] et E[Y|X, D=1].
    L'effet est heterogene en X1 mais E[X1] = 0, donc l'ATE vaut theta."""
    if design == "linear":
        score = 0.6 * X[:, 0] - 0.4 * X[:, 1]
        g0 = X[:, 0] + 0.5 * X[:, 2] - 0.4 * X[:, 3]
    else:
        score = 0.7 * X[:, 0] + 0.5 * np.tanh(X[:, 1]) - 0.3 * X[:, 2] * X[:, 3]
        g0 = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.5 * X[:, 2] * (X[:, 3] > 0)
    g1 = g0 + theta + 0.3 * X[:, 1]
    return expit(score), g0, g1


def dgp_irm(n, p, theta, rng, design="complex"):
    X = draw_x(n, p, rng)
    prop, g0, g1 = irm_nuisance(X, theta, design)
    D = rng.binomial(1, prop).astype(float)
    Y = np.where(D == 1, g1, g0) + rng.normal(size=n)
    return Y, D, X
