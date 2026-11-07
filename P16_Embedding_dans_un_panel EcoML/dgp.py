import numpy as np
import pandas as pd


def simulate_panel(scenario="additive", n_units=300, n_times=10, beta=1.0,
                   obs_prob=1.0, rank=2, seed=0):
    """Panel simule. Le regresseur x est correle aux effets individuels.
    - additive : y = beta x + alpha_i + gamma_t + e
    - interactive : on ajoute une composante lambda_i' f_t, correlee a x, que les effets
      fixes additifs ne peuvent pas absorber.
    obs_prob < 1 donne un panel desequilibre (chaque unite garde au moins une observation)."""
    rng = np.random.default_rng(seed)
    alpha = rng.normal(size=n_units)
    gamma = 0.5 * rng.normal(size=n_times)
    lam = rng.normal(size=(n_units, rank))
    f = rng.normal(size=(n_times, rank))
    common = lam @ f.T

    x = 0.8 * alpha[:, None] + 0.5 * gamma[None, :] + rng.normal(size=(n_units, n_times))
    y = beta * x + alpha[:, None] + gamma[None, :] + rng.normal(size=(n_units, n_times))
    if scenario == "interactive":
        x = x + 0.6 * common
        y = beta * x + alpha[:, None] + gamma[None, :] + common + rng.normal(size=(n_units, n_times))

    observed = rng.random((n_units, n_times)) < obs_prob
    for i in np.flatnonzero(~observed.any(axis=1)):
        observed[i, rng.integers(n_times)] = True
    u, t = np.nonzero(observed)
    return pd.DataFrame({"unit": u, "time": t, "x": x[u, t], "y": y[u, t]})


def split_cells(df, test_share=0.2, seed=0):
    rng = np.random.default_rng(seed)
    test = rng.random(len(df)) < test_share
    return df[~test].reset_index(drop=True), df[test].reset_index(drop=True)
