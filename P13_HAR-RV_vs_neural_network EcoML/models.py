import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Chaque modele a fit(data, idx) et predict(data, idx).
# data contient X (DataFrame des variables), seq (tableau jours x 22 x 3) et y (cible).


class LinearModel:
    def __init__(self, cols):
        self.cols = cols

    def _design(self, data, idx):
        return np.column_stack([np.ones(len(idx)), data.X[self.cols].values[idx]])

    def fit(self, data, idx):
        self.coef = np.linalg.lstsq(self._design(data, idx), data.y[idx], rcond=None)[0]
        return self

    def predict(self, data, idx):
        return self._design(data, idx) @ self.coef


class RidgeLags:
    """Ridge sur les 22 derniers log(RV) quotidiens, sans imposer la structure HAR."""

    def fit(self, data, idx):
        self.mu = data.seq[idx, :, 0].mean(0)
        self.model = RidgeCV(alphas=np.logspace(-3, 3, 20)).fit(data.seq[idx, :, 0] - self.mu, data.y[idx])
        return self

    def predict(self, data, idx):
        return self.model.predict(data.seq[idx, :, 0] - self.mu)


class ForestModel:
    def __init__(self, cols, n_estimators=200, min_samples_leaf=10, seed=0):
        self.cols = cols
        self.model = RandomForestRegressor(n_estimators=n_estimators,
                                           min_samples_leaf=min_samples_leaf, random_state=seed)

    def fit(self, data, idx):
        self.model.fit(data.X[self.cols].values[idx], data.y[idx])
        return self

    def predict(self, data, idx):
        return self.model.predict(data.X[self.cols].values[idx])


if HAS_TORCH:
    def _train_net(net, Xtr, ytr, Xva, yva, epochs, lr, weight_decay, patience):
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        best, best_state, wait = np.inf, None, 0
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(Xtr))
            for s in range(0, len(Xtr), 64):
                b = perm[s:s + 64]
                opt.zero_grad()
                loss = ((net(Xtr[b]).squeeze(-1) - ytr[b]) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = ((net(Xva).squeeze(-1) - yva) ** 2).mean().item()
            if val < best - 1e-6:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net.eval()

    class _LSTMNet(nn.Module):
        def __init__(self, d_in, hidden=16):
            super().__init__()
            self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            return self.out(h[:, -1])

    class _NetModel:
        """Base commune : standardisation, validation en fin de fenetre, moyenne de plusieurs graines."""

        def __init__(self, epochs=150, lr=1e-3, weight_decay=1e-4, patience=15, n_seeds=3):
            self.epochs = epochs
            self.lr = lr
            self.weight_decay = weight_decay
            self.patience = patience
            self.n_seeds = n_seeds

        def _inputs(self, data, idx):
            raise NotImplementedError

        def _make_net(self, d_in):
            raise NotImplementedError

        def fit(self, data, idx):
            X = self._inputs(data, idx)
            y = data.y[idx]
            axes = tuple(range(X.ndim - 1))
            self.mu, self.sd = X.mean(axes), X.std(axes) + 1e-8
            self.ym, self.ys = y.mean(), y.std() + 1e-8
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            yt = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)
            n_val = max(int(0.15 * len(yt)), 1)
            self.nets = []
            for seed in range(self.n_seeds):
                torch.manual_seed(seed)
                net = self._make_net(X.shape[-1])
                net = _train_net(net, Xt[:-n_val], yt[:-n_val], Xt[-n_val:], yt[-n_val:],
                                 self.epochs, self.lr, self.weight_decay, self.patience)
                self.nets.append(net)
            return self

        def predict(self, data, idx):
            X = self._inputs(data, idx)
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            with torch.no_grad():
                preds = [net(Xt).squeeze(-1).numpy() for net in self.nets]
            return np.mean(preds, axis=0) * self.ys + self.ym

    class MLPModel(_NetModel):
        def __init__(self, cols, hidden=32, **kw):
            super().__init__(**kw)
            self.cols = cols
            self.hidden = hidden

        def _inputs(self, data, idx):
            return data.X[self.cols].values[idx]

        def _make_net(self, d_in):
            h = self.hidden
            return nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    class LSTMModel(_NetModel):
        def __init__(self, hidden=16, **kw):
            super().__init__(**kw)
            self.hidden = hidden

        def _inputs(self, data, idx):
            return data.seq[idx]

        def _make_net(self, d_in):
            return _LSTMNet(d_in, self.hidden)
