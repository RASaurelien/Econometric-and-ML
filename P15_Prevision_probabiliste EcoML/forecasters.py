import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def make_lags(y, p):
    """Ligne t : (y[t-1], ..., y[t-p]), cible y[t], pour t >= p."""
    X = np.column_stack([y[p - k - 1: len(y) - k - 1] for k in range(p)])
    return X, y[p:]


class ARForecaster:
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        self.coef = np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X]) @ self.coef


class GBMForecaster:
    def __init__(self, seed=0):
        self.model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                               subsample=0.8, random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class QuantileGBM:
    """Deux regressions quantiles (bas et haut) pour la regression quantile conforme."""

    def __init__(self, alpha, seed=0):
        make = lambda q: GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=150, max_depth=3,
                                                   learning_rate=0.05, subsample=0.8, random_state=seed)
        self.lo = make(alpha / 2)
        self.hi = make(1 - alpha / 2)

    def fit(self, X, y):
        self.lo.fit(X, y)
        self.hi.fit(X, y)
        return self

    def predict(self, X):
        lo, hi = self.lo.predict(X), self.hi.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)
