import numpy as np
from scipy.special import expit


def cate_function(X):
    return 0.5 + np.maximum(X[:, 0], 0) + 0.8 * np.sin(2 * X[:, 1]) - 0.7 * (X[:, 2] > 0.5)


def baseline(X):
    return X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.3 * X[:, 3] ** 2


def simulate(n, p=10, rng=None, randomized=True, noise=1.0):
    rng = np.random.default_rng() if rng is None else rng
    X = rng.normal(size=(n, p))
    if randomized:
        e = np.full(n, 0.5)
    else:
        e = expit(0.8 * X[:, 0] - 0.5 * X[:, 3])
    T = rng.binomial(1, e)
    tau = cate_function(X)
    Y = baseline(X) + T * tau + noise * rng.normal(size=n)
    return {"X": X, "T": T, "Y": Y, "tau": tau, "e": e}
