import warnings
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TorchMLP(BaseEstimator):
    """Petit reseau de neurones utilise comme learner de nuisance."""

    def __init__(self, task="reg", hidden=64, epochs=200, lr=1e-3,
                 weight_decay=1e-3, patience=15, seed=0):
        self.task = task
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)
        y = np.asarray(y, dtype=np.float32)
        if self.task == "reg":
            self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() + 1e-8)
            y = (y - self.y_mean_) / self.y_std_

        idx = np.random.default_rng(self.seed).permutation(len(y))
        n_val = max(int(0.15 * len(y)), 1)
        va, tr = idx[:n_val], idx[n_val:]
        Xtr, ytr = torch.from_numpy(Xs[tr]), torch.from_numpy(y[tr])
        Xva, yva = torch.from_numpy(Xs[va]), torch.from_numpy(y[va])

        h = self.hidden
        self.net_ = nn.Sequential(nn.Linear(Xs.shape[1], h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        loss_fn = nn.MSELoss() if self.task == "reg" else nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(tr))
            for s in range(0, len(tr), 64):
                b = perm[s:s + 64]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xtr[b]).squeeze(1), ytr[b])
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                val = loss_fn(self.net_(Xva).squeeze(1), yva).item()
            if val < best - 1e-5:
                best = val
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def _forward(self, X):
        Xs = torch.from_numpy(self.scaler_.transform(X).astype(np.float32))
        self.net_.eval()
        with torch.no_grad():
            return self.net_(Xs).squeeze(1).numpy()

    def predict(self, X):
        out = self._forward(X)
        if self.task == "reg":
            return out * self.y_std_ + self.y_mean_
        return 1 / (1 + np.exp(-out))

    def predict_proba(self, X):
        p = self.predict(X)
        return np.column_stack([1 - p, p])


def make_learner(name, task="reg"):
    if task == "reg":
        options = {
            "lasso": lambda: LassoCV(cv=5),
            "rf": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5),
            "gbm": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=3,
                                                     learning_rate=0.05, subsample=0.8),
            "mlp": lambda: TorchMLP("reg"),
        }
    else:
        options = {
            "lasso": lambda: LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=8, cv=5),
            "rf": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=5),
            "gbm": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                                      learning_rate=0.05, subsample=0.8),
            "mlp": lambda: TorchMLP("clf"),
        }
    return options[name]()


def _plr_once(Y, D, X, learner, K, seed):
    n = len(Y)
    res_y, res_d = np.zeros(n), np.zeros(n)
    for tr, te in KFold(K, shuffle=True, random_state=seed).split(X):
        res_y[te] = Y[te] - make_learner(learner).fit(X[tr], Y[tr]).predict(X[te])
        res_d[te] = D[te] - make_learner(learner).fit(X[tr], D[tr]).predict(X[te])
    theta = (res_d @ res_y) / (res_d @ res_d)
    psi = (res_y - theta * res_d) * res_d
    se = np.sqrt(np.mean(psi ** 2) / np.mean(res_d ** 2) ** 2 / n)
    return theta, se


def dml_plr(Y, D, X, learner="rf", K=5, n_rep=1, seed=0):
    """Modele partiellement lineaire, cross-fitting repete (mediane des estimations)."""
    fits = np.array([_plr_once(Y, D, X, learner, K, seed + r) for r in range(n_rep)])
    theta = np.median(fits[:, 0])
    se = np.sqrt(np.median(fits[:, 1] ** 2 + (fits[:, 0] - theta) ** 2))
    return theta, se


def dml_irm(Y, D, X, learner="rf", K=5, trim=0.02, seed=0):
    """Traitement binaire, score AIPW avec cross-fitting. Renvoie l'ATE."""
    n = len(Y)
    g0, g1, m = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in StratifiedKFold(K, shuffle=True, random_state=seed).split(X, D):
        t0, t1 = tr[D[tr] == 0], tr[D[tr] == 1]
        g0[te] = make_learner(learner).fit(X[t0], Y[t0]).predict(X[te])
        g1[te] = make_learner(learner).fit(X[t1], Y[t1]).predict(X[te])
        m[te] = make_learner(learner, "clf").fit(X[tr], D[tr]).predict_proba(X[te])[:, 1]
    m = np.clip(m, trim, 1 - trim)
    psi = g1 - g0 + D * (Y - g1) / m - (1 - D) * (Y - g0) / (1 - m)
    return psi.mean(), psi.std(ddof=1) / np.sqrt(n)


def _ols(Y, D, X=None):
    Z = D.reshape(-1, 1) if X is None else np.column_stack([D, X])
    fit = sm.OLS(Y, sm.add_constant(Z)).fit(cov_type="HC1")
    return fit.params[1], fit.bse[1]


def ols_naive(Y, D, X):
    return _ols(Y, D)


def ols_full(Y, D, X):
    if X.shape[1] >= len(Y) - 10:
        return np.nan, np.nan
    return _ols(Y, D, X)


def post_lasso(Y, D, X):
    lasso = LassoCV(cv=5).fit(np.column_stack([D, X]), Y)
    keep = np.flatnonzero(lasso.coef_[1:])
    return _ols(Y, D, X[:, keep] if len(keep) else None)


def post_double_selection(Y, D, X):
    sel_y = np.flatnonzero(LassoCV(cv=5).fit(X, Y).coef_)
    sel_d = np.flatnonzero(LassoCV(cv=5).fit(X, D).coef_)
    keep = np.union1d(sel_y, sel_d).astype(int)
    return _ols(Y, D, X[:, keep] if len(keep) else None)


def confint(theta, se, level=0.95):
    z = norm.ppf(0.5 + level / 2)
    return theta - z * se, theta + z * se
