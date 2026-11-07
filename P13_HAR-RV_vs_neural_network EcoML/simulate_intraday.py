import numpy as np
import pandas as pd


def simulate_returns(n_days=3000, m=78, seed=0, jump_prob=0.05, leverage=0.35):
    """Rendements intraday simules : log-variance a deux composantes persistantes
    (memoire longue approchee), saisonnalite en U, effet de levier et sauts."""
    rng = np.random.default_rng(seed)
    f1 = f2 = 0.0
    phi1, phi2, s1, s2 = 0.985, 0.6, 0.08, 0.30
    j = np.arange(m)
    shape = 1 + 2.0 * np.abs((j - m / 2) / (m / 2)) ** 2
    shape = shape / shape.sum()

    returns = np.zeros((n_days, m))
    prev_z = 0.0
    for t in range(n_days):
        f1 = phi1 * f1 + s1 * (-leverage * prev_z + np.sqrt(1 - leverage ** 2) * rng.normal())
        f2 = phi2 * f2 + s2 * rng.normal()
        var = np.exp(-9.3 + f1 + f2)
        r = np.sqrt(var * shape) * rng.normal(size=m)
        if rng.random() < jump_prob:
            r[rng.integers(m)] += np.sqrt(4 * var) * rng.normal()
        returns[t] = r
        prev_z = r.sum() / np.sqrt(var)
    idx = pd.bdate_range("2005-01-03", periods=n_days)
    return returns, idx
