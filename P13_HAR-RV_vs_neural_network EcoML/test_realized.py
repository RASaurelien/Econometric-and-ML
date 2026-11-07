import os
import tempfile
import numpy as np
import pandas as pd

import realized as R
from simulate_intraday import simulate_returns


def test_measures_add_up():
    r = np.random.default_rng(0).normal(scale=0.001, size=78)
    out = R.measures_from_returns(r)
    assert abs(out["rv"] - np.sum(r ** 2)) < 1e-15
    assert abs(out["rsv_pos"] + out["rsv_neg"] - out["rv"]) < 1e-15
    assert out["jump"] >= 0


def test_bipower_close_to_rv_without_jumps():
    ret, _ = simulate_returns(300, jump_prob=0.0, seed=1)
    m = R.measures_from_array(ret)
    assert abs(m["bpv"].sum() / m["rv"].sum() - 1) < 0.1


def test_csv_loader_matches_arrays():
    ret, idx = simulate_returns(20, seed=3)
    rows = []
    for d, r in zip(idx, ret):
        times = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=ret.shape[1] + 1, freq="5min")
        logp = np.concatenate([[np.log(100)], np.log(100) + np.cumsum(r)])
        rows.append(pd.DataFrame({"timestamp": times, "price": np.exp(logp)}))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prix.csv")
        pd.concat(rows).to_csv(path, index=False)
        from_csv = R.measures_from_csv(path)
    from_arr = R.measures_from_array(ret, idx)
    assert np.abs(from_csv.values - from_arr.values).max() < 1e-12


def test_target_is_future_average():
    m = pd.DataFrame({"rv": np.arange(1.0, 11.0)})
    y = R.make_target(m, h=3)
    assert abs(y[0] - np.log((2 + 3 + 4) / 3)) < 1e-12
    assert np.isnan(y[-1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
