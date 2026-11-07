import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def default_reg():
    return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8)


class SLearner:
    def __init__(self, base=None):
        self.base = base or default_reg()

    def fit(self, X, T, Y):
        self.model = clone(self.base).fit(np.column_stack([X, T]), Y)
        return self

    def predict(self, X):
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        return self.model.predict(np.column_stack([X, ones])) - self.model.predict(np.column_stack([X, zeros]))


class TLearner:
    def __init__(self, base=None):
        self.base = base or default_reg()

    def fit(self, X, T, Y):
        self.m0 = clone(self.base).fit(X[T == 0], Y[T == 0])
        self.m1 = clone(self.base).fit(X[T == 1], Y[T == 1])
        return self

    def predict(self, X):
        return self.m1.predict(X) - self.m0.predict(X)


class XLearner:
    def __init__(self, base=None):
        self.base = base or default_reg()

    def fit(self, X, T, Y):
        self.m0 = clone(self.base).fit(X[T == 0], Y[T == 0])
        self.m1 = clone(self.base).fit(X[T == 1], Y[T == 1])
        d1 = Y[T == 1] - self.m0.predict(X[T == 1])
        d0 = self.m1.predict(X[T == 0]) - Y[T == 0]
        self.tau1 = clone(self.base).fit(X[T == 1], d1)
        self.tau0 = clone(self.base).fit(X[T == 0], d0)
        self.e = GradientBoostingClassifier(n_estimators=100, max_depth=2).fit(X, T)
        return self

    def predict(self, X):
        g = self.e.predict_proba(X)[:, 1]
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)


class DRLearner:
    def __init__(self, base=None, K=5):
        self.base = base or default_reg()
        self.K = K

    def fit(self, X, T, Y):
        n = len(Y)
        mu0, mu1, e = np.zeros(n), np.zeros(n), np.zeros(n)
        for tr, te in KFold(self.K, shuffle=True, random_state=0).split(X):
            t0, t1 = tr[T[tr] == 0], tr[T[tr] == 1]
            mu0[te] = clone(self.base).fit(X[t0], Y[t0]).predict(X[te])
            mu1[te] = clone(self.base).fit(X[t1], Y[t1]).predict(X[te])
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=2).fit(X[tr], T[tr])
            e[te] = clf.predict_proba(X[te])[:, 1]
        e = np.clip(e, 0.05, 0.95)
        pseudo = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
        self.final = clone(self.base).fit(X, pseudo)
        return self

    def predict(self, X):
        return self.final.predict(X)


class HonestCausalForest:
    """Foret causale simplifiee : on construit les arbres sur une moitie
    de l'echantillon (outcome transforme) et on estime les effets dans les
    feuilles avec l'autre moitie (honnetete)."""

    def __init__(self, n_trees=200, min_leaf=20, max_features=0.6, e=0.5, seed=0):
        self.n_trees = n_trees
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.e = e
        self.seed = seed

    def fit(self, X, T, Y):
        rng = np.random.default_rng(self.seed)
        n = len(Y)
        ystar = Y * (T - self.e) / (self.e * (1 - self.e))
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.permutation(n)
            half = len(idx) // 2
            a, b = idx[:half], idx[half:]
            tree = DecisionTreeRegressor(min_samples_leaf=self.min_leaf,
                                         max_features=self.max_features,
                                         random_state=int(rng.integers(1e9)))
            tree.fit(X[a], ystar[a])
            leaves = tree.apply(X[b])
            global_tau = Y[b][T[b] == 1].mean() - Y[b][T[b] == 0].mean()
            tau_leaf = {}
            for leaf in np.unique(leaves):
                m = leaves == leaf
                tb, yb = T[b][m], Y[b][m]
                if (tb == 1).sum() >= 3 and (tb == 0).sum() >= 3:
                    tau_leaf[leaf] = yb[tb == 1].mean() - yb[tb == 0].mean()
                else:
                    tau_leaf[leaf] = global_tau
            self.trees.append((tree, tau_leaf, global_tau))
        return self

    def predict(self, X):
        out = np.zeros(len(X))
        for tree, tau_leaf, g in self.trees:
            leaves = tree.apply(X)
            out += np.array([tau_leaf.get(l, g) for l in leaves])
        return out / len(self.trees)


if HAS_TORCH:
    class TARNet:
        """Representation partagee puis deux tetes (controle / traite)."""

        def __init__(self, hidden=64, epochs=150, lr=1e-3, weight_decay=1e-4, seed=0):
            self.hidden = hidden
            self.epochs = epochs
            self.lr = lr
            self.weight_decay = weight_decay
            self.seed = seed

        def fit(self, X, T, Y):
            torch.manual_seed(self.seed)
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
            self.ym, self.ys = Y.mean(), Y.std()
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            Tt = torch.tensor(T, dtype=torch.float32)
            Yt = torch.tensor((Y - self.ym) / self.ys, dtype=torch.float32)
            h = self.hidden
            self.trunk = nn.Sequential(nn.Linear(X.shape[1], h), nn.ELU(), nn.Linear(h, h), nn.ELU())
            self.head0 = nn.Sequential(nn.Linear(h, h // 2), nn.ELU(), nn.Linear(h // 2, 1))
            self.head1 = nn.Sequential(nn.Linear(h, h // 2), nn.ELU(), nn.Linear(h // 2, 1))
            params = list(self.trunk.parameters()) + list(self.head0.parameters()) + list(self.head1.parameters())
            opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
            for _ in range(self.epochs):
                perm = torch.randperm(len(Yt))
                for s in range(0, len(Yt), 128):
                    b = perm[s:s + 128]
                    opt.zero_grad()
                    z = self.trunk(Xt[b])
                    pred = torch.where(Tt[b] == 1, self.head1(z).squeeze(1), self.head0(z).squeeze(1))
                    loss = ((pred - Yt[b]) ** 2).mean()
                    loss.backward()
                    opt.step()
            return self

        def predict(self, X):
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            with torch.no_grad():
                z = self.trunk(Xt)
                return ((self.head1(z) - self.head0(z)).squeeze(1) * self.ys).numpy()
