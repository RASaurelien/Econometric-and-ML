import numpy as np
import pandas as pd

MU1_SQ = 2 / np.pi


def measures_from_returns(r):
    """Mesures de volatilite realisee d'une journee a partir des rendements intraday."""
    r = np.asarray(r, dtype=float)
    rv = np.sum(r ** 2)
    rsv_pos = np.sum(np.where(r > 0, r, 0.0) ** 2)
    rsv_neg = np.sum(np.where(r < 0, r, 0.0) ** 2)
    bpv = np.sum(np.abs(r[1:]) * np.abs(r[:-1])) / MU1_SQ
    jump = max(rv - bpv, 0.0)
    return {"rv": rv, "rsv_pos": rsv_pos, "rsv_neg": rsv_neg, "bpv": bpv, "jump": jump}


def measures_from_array(returns, index=None):
    """returns : tableau (jours, rendements intraday)."""
    rows = [measures_from_returns(day) for day in returns]
    return pd.DataFrame(rows, index=index)


def measures_from_csv(path, freq="5min"):
    """Fichier avec une colonne de dates et heures (premiere colonne) et une colonne `price`."""
    px = pd.read_csv(path, parse_dates=[0], index_col=0)["price"].sort_index()
    px = px.resample(freq).last().dropna()
    logp = np.log(px)
    day = logp.index.normalize()
    r = logp.groupby(day).diff().dropna()
    out = r.groupby(r.index.normalize()).apply(lambda s: pd.Series(measures_from_returns(s.values)))
    return out.unstack() if isinstance(out, pd.Series) else out


def build_features(m, seq_len=22):
    """Variables HAR (en logs) et sequences pour les reseaux de neurones."""
    feats = pd.DataFrame(index=m.index)
    feats["d"] = np.log(m["rv"])
    feats["w"] = np.log(m["rv"].rolling(5).mean())
    feats["m"] = np.log(m["rv"].rolling(22).mean())
    feats["rsv_neg"] = np.log(m["rsv_neg"] + 1e-12)
    feats["rsv_pos"] = np.log(m["rsv_pos"] + 1e-12)
    feats["bpv"] = np.log(m["bpv"] + 1e-12)

    base = np.column_stack([feats["d"], feats["rsv_neg"], feats["rsv_pos"]])
    seq = np.full((len(m), seq_len, base.shape[1]), np.nan)
    for t in range(seq_len - 1, len(m)):
        seq[t] = base[t - seq_len + 1: t + 1]
    return feats, seq


def make_target(m, h=1):
    """log de la moyenne de RV sur les h jours suivants."""
    fut = m["rv"].rolling(h).mean().shift(-h)
    return np.log(fut).values
