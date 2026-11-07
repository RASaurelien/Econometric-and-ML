import numpy as np
import pandas as pd


def transform_series(x, code):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        if code == 1:
            out = x
        elif code == 2:
            out[1:] = x[1:] - x[:-1]
        elif code == 3:
            out[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
        elif code == 4:
            out = np.log(x)
        elif code == 5:
            out[1:] = np.log(x[1:]) - np.log(x[:-1])
        elif code == 6:
            lx = np.log(x)
            out[2:] = lx[2:] - 2 * lx[1:-1] + lx[:-2]
        elif code == 7:
            r = x[1:] / x[:-1] - 1
            out[2:] = r[1:] - r[:-1]
    return out


def load_fred_md(path):
    raw = pd.read_csv(path)
    tcodes = raw.iloc[0, 1:].astype(float)
    df = raw.iloc[1:].copy()
    df.index = pd.to_datetime(df["sasdate"], format="%m/%d/%Y")
    df = df.drop(columns="sasdate").astype(float)
    out = pd.DataFrame({c: transform_series(df[c].values, int(tcodes[c])) for c in df.columns},
                       index=df.index)
    out = out.iloc[2:]
    # valeurs extremes : on borne a 10 ecarts interquartiles autour de la mediane
    med, iqr = out.median(), out.quantile(0.75) - out.quantile(0.25)
    out = out.clip(med - 10 * iqr, med + 10 * iqr, axis=1)
    out = out.dropna(axis=1, thresh=int(0.8 * len(out)))
    # remplissage causal : derniere valeur connue, puis 0 pour les debuts de serie manquants
    return out.ffill().fillna(0.0)


def make_synthetic(T=600, N=80, r=3, seed=0):
    # donnees de test : modele a facteurs avec cible partiellement non lineaire
    rng = np.random.default_rng(seed)
    F = np.zeros((T, r))
    for t in range(1, T):
        F[t] = 0.7 * F[t - 1] + rng.normal(size=r)
    L = rng.normal(size=(N, r))
    X = F @ L.T + rng.normal(size=(T, N))
    target = np.zeros(T)
    for t in range(1, T):
        target[t] = 0.3 * target[t - 1] + 0.5 * F[t - 1, 0] + 0.3 * np.tanh(2 * F[t - 1, 1]) + 0.5 * rng.normal()
    cols = ["TARGET"] + [f"x{i}" for i in range(N - 1)]
    X[:, 0] = target
    idx = pd.date_range("1970-01-01", periods=T, freq="MS")
    return pd.DataFrame(X, index=idx, columns=cols)
