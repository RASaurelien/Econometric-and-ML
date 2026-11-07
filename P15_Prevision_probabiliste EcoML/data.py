import numpy as np
import pandas as pd


def simulate_series(T=4000, seed=0, shift=True):
    """AR non lineaire avec volatilite a deux regimes (chaine de Markov persistante),
    erreurs de Student a 5 degres de liberte, et une rupture a 70 % de l'echantillon
    (volatilite multipliee et dynamique modifiee) si shift=True."""
    rng = np.random.default_rng(seed)
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        stay = 0.985
        regime[t] = regime[t - 1] if rng.random() < stay else 1 - regime[t - 1]
    sigma = np.where(regime == 0, 0.5, 1.4)
    eps = rng.standard_t(5, size=T) / np.sqrt(5 / 3)
    brk = int(0.7 * T)
    y = np.zeros(T)
    for t in range(2, T):
        if shift and t >= brk:
            mean = 0.3 * y[t - 1] + 0.4 * y[t - 2]
            scale = 1.8 * sigma[t]
        else:
            mean = 0.6 * y[t - 1] - 0.3 * y[t - 2] + 0.8 * np.tanh(y[t - 1])
            scale = sigma[t]
        y[t] = mean + scale * eps[t]
    return pd.DataFrame({"y": y, "regime": regime})


def load_series(path, column):
    s = pd.read_csv(path)[column].astype(float).dropna().values
    return pd.DataFrame({"y": s, "regime": np.zeros(len(s), dtype=int)})
