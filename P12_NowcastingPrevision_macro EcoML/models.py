import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# interface commune : fit(X_hist, y_hist) puis predict(X_hist)
# X_hist contient toutes les lignes jusqu'a la date de prevision, y_hist la cible
# a h pas deja realisee (donc plus courte). On prevoit y pour la derniere ligne de X_hist.


class AR:
    def __init__(self, target_col, max_lag=6):
        self.col = target_col
        self.max_lag = max_lag

    def _design(self, z, p):
        rows = np.column_stack([z[p - k - 1: len(z) - k - 1] for k in range(p)])
        return np.column_stack([np.ones(len(rows)), rows]), z[p:]

    def fit(self, X, y):
        z = X[:, self.col]
        best = None
        for p in range(1, self.max_lag + 1):
            A, b = self._design(z[: len(y)], p)
            coef = np.linalg.lstsq(A, b, rcond=None)[0]
            rss = np.sum((b - A @ coef) ** 2)
            bic = len(b) * np.log(rss / len(b)) + (p + 1) * np.log(len(b))
            if best is None or bic < best[0]:
                best = (bic, p)
        self.p = best[1]
        # on regresse la cible a h pas sur les p retards de la variable cible
        lags = np.column_stack([np.ones(len(y) - self.p + 1)] +
                               [z[self.p - 1 - k: len(y) - k] for k in range(self.p)])
        self.coef = np.linalg.lstsq(lags, y[self.p - 1:], rcond=None)[0]
        return self

    def predict(self, X):
        z = X[:, self.col]
        last = np.concatenate([[1.0], z[::-1][: self.p]])
        return float(last @ self.coef)


class DiffusionIndex:
    """Stock et Watson : facteurs principaux puis regression lineaire."""

    def __init__(self, n_factors=8, lags=2):
        self.r = n_factors
        self.lags = lags

    def _features(self, F):
        cols = [F]
        for k in range(1, self.lags):
            shifted = np.vstack([np.zeros((k, F.shape[1])), F[:-k]])
            cols.append(shifted)
        return np.column_stack(cols)

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(self.r).fit(self.scaler.transform(X))
        F = self.pca.transform(self.scaler.transform(X))
        Z = np.column_stack([np.ones(len(F)), self._features(F)])
        self.coef = np.linalg.lstsq(Z[: len(y)], y, rcond=None)[0]
        return self

    def predict(self, X):
        F = self.pca.transform(self.scaler.transform(X))
        Z = np.column_stack([np.ones(len(F)), self._features(F)])
        return float(Z[-1] @ self.coef)


class ElasticNetModel:
    def __init__(self, n_lags=2):
        self.n_lags = n_lags

    def _features(self, X):
        cols = [X]
        for k in range(1, self.n_lags):
            cols.append(np.vstack([np.zeros((k, X.shape[1])), X[:-k]]))
        return np.column_stack(cols)

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Z = self._features(self.scaler.transform(X))[: len(y)]
        self.model = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.9], cv=TimeSeriesSplit(4),
                                  max_iter=3000).fit(Z, y)
        return self

    def predict(self, X):
        Z = self._features(self.scaler.transform(X))
        return float(self.model.predict(Z[-1:])[0])


if HAS_TORCH:
    class _LSTMNet(nn.Module):
        def __init__(self, d_in, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            return self.out(h[:, -1]).squeeze(1)

    class _TransformerNet(nn.Module):
        def __init__(self, d_in, d_model=32, window=12):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model)
            self.pos = nn.Parameter(torch.zeros(1, window, d_model))
            layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=64,
                                               dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=1)
            self.out = nn.Linear(d_model, 1)

        def forward(self, x):
            h = self.enc(self.proj(x) + self.pos)
            return self.out(h[:, -1]).squeeze(1)

    class SequenceModel:
        """LSTM ou transformer sur une fenetre de facteurs principaux."""

        def __init__(self, kind="lstm", n_factors=10, window=12, epochs=60,
                     lr=2e-3, n_seeds=3):
            self.kind = kind
            self.r = n_factors
            self.window = window
            self.epochs = epochs
            self.lr = lr
            self.n_seeds = n_seeds

        def _windows(self, F):
            pad = np.vstack([np.zeros((self.window - 1, F.shape[1])), F])
            return np.stack([pad[i: i + self.window] for i in range(len(F))])

        def _net(self):
            if self.kind == "lstm":
                return _LSTMNet(self.r)
            return _TransformerNet(self.r, window=self.window)

        def fit(self, X, y):
            self.scaler = StandardScaler().fit(X)
            self.pca = PCA(self.r).fit(self.scaler.transform(X))
            F = self.pca.transform(self.scaler.transform(X))
            self.fscale = F.std(0)
            F = F / self.fscale
            W = torch.tensor(self._windows(F)[: len(y)], dtype=torch.float32)
            self.ym, self.ys = y.mean(), y.std() + 1e-8
            Y = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)
            self.nets = []
            for seed in range(self.n_seeds):
                torch.manual_seed(seed)
                net = self._net()
                opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-3)
                for _ in range(self.epochs):
                    net.train()
                    perm = torch.randperm(len(Y))
                    for s in range(0, len(Y), 64):
                        b = perm[s:s + 64]
                        opt.zero_grad()
                        loss = ((net(W[b]) - Y[b]) ** 2).mean()
                        loss.backward()
                        opt.step()
                self.nets.append(net.eval())
            return self

        def predict(self, X):
            F = self.pca.transform(self.scaler.transform(X)) / self.fscale
            W = torch.tensor(self._windows(F)[-1:], dtype=torch.float32)
            with torch.no_grad():
                preds = [net(W).item() for net in self.nets]
            return float(np.mean(preds) * self.ys + self.ym)
