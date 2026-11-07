import numpy as np

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class PanelNet(nn.Module):
        """y = intercept + z'coef + a_i + g_t (+ <u_i, v_t> si interactive).
        z contient x en premiere colonne, et eventuellement les moyennes de Mundlak.
        Les effets sont des embeddings appris avec une penalite L2 (equivalent d'un effet aleatoire)."""

        def __init__(self, n_units, n_times, n_cov=1, dim=4, interactive=False):
            super().__init__()
            self.interactive = interactive
            self.cov = nn.Linear(n_cov, 1)
            nn.init.zeros_(self.cov.weight)
            nn.init.zeros_(self.cov.bias)
            self.unit_fe = nn.Embedding(n_units, 1)
            self.time_fe = nn.Embedding(n_times, 1)
            nn.init.zeros_(self.unit_fe.weight)
            nn.init.zeros_(self.time_fe.weight)
            if interactive:
                self.unit_emb = nn.Embedding(n_units, dim)
                self.time_emb = nn.Embedding(n_times, dim)
                nn.init.normal_(self.unit_emb.weight, std=0.1)
                nn.init.normal_(self.time_emb.weight, std=0.1)

        def forward(self, z, u, t):
            out = self.cov(z).squeeze(-1) + self.unit_fe(u).squeeze(-1) + self.time_fe(t).squeeze(-1)
            if self.interactive:
                out = out + (self.unit_emb(u) * self.time_emb(t)).sum(-1)
            return out

        def penalty(self):
            p = (self.unit_fe.weight ** 2).sum() + (self.time_fe.weight ** 2).sum()
            if self.interactive:
                p = p + (self.unit_emb.weight ** 2).sum() + (self.time_emb.weight ** 2).sum()
            return p

    def _tensors(df, cols):
        z = torch.tensor(df[cols].values, dtype=torch.float32)
        y = torch.tensor(df["y"].values, dtype=torch.float32)
        u = torch.tensor(df["unit"].values, dtype=torch.long)
        t = torch.tensor(df["time"].values, dtype=torch.long)
        return z, y, u, t

    def fit_embedding(train, n_units, n_times, covariates=("x",), interactive=False, dim=4,
                      lambdas=(0.3, 1.0, 3.0), epochs=600, lr=0.03, val_share=0.1, seed=0):
        """Entraine le reseau en batch complet avec Adam. La penalite et le nombre d'epoques
        sont choisis sur un echantillon de validation tire parmi les cellules d'entrainement."""
        cols = list(covariates)
        z, y, u, t = _tensors(train, cols)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train))
        n_val = max(int(val_share * len(train)), 1)
        va = torch.tensor(perm[:n_val], dtype=torch.long)
        tr = torch.tensor(perm[n_val:], dtype=torch.long)

        best_val, best_state, best_lam = np.inf, None, None
        for lam in lambdas:
            torch.manual_seed(seed)
            net = PanelNet(n_units, n_times, len(cols), dim, interactive)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            for epoch in range(epochs):
                opt.zero_grad()
                pred = net(z[tr], u[tr], t[tr])
                loss = ((pred - y[tr]) ** 2).mean() + lam * net.penalty() / len(tr)
                loss.backward()
                opt.step()
                if epoch % 10 == 9:
                    with torch.no_grad():
                        val = ((net(z[va], u[va], t[va]) - y[va]) ** 2).mean().item()
                    if val < best_val:
                        best_val, best_lam = val, lam
                        best_state = {k: v.clone() for k, v in net.state_dict().items()}

        net = PanelNet(n_units, n_times, len(cols), dim, interactive)
        net.load_state_dict(best_state)
        net.eval()

        def predict(df):
            zz, _, uu, tt = _tensors(df, cols)
            with torch.no_grad():
                return net(zz, uu, tt).numpy()

        return net.cov.weight[0, 0].item(), predict, best_lam
