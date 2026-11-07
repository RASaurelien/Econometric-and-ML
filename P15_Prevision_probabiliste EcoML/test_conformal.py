import numpy as np

import conformal as C
from evaluate import rolling_coverage


def test_conformal_quantile_finite_sample():
    scores = np.arange(1.0, 10.0)
    assert C.conformal_quantile(scores, 0.9) == 9.0
    assert C.conformal_quantile(scores, 0.5) == 5.0
    assert C.conformal_quantile(scores, 0.99) == np.inf
    assert C.conformal_quantile(scores, 0.0) == -np.inf


def test_split_conformal_coverage_exchangeable():
    rng = np.random.default_rng(0)
    cal = np.abs(rng.standard_t(5, size=20000))
    new = rng.standard_t(5, size=200000)
    lo, hi = C.split_conformal(np.zeros(len(new)), np.zeros(len(new)), cal, 0.1)
    cover = np.mean((lo <= new) & (new <= hi))
    assert abs(cover - 0.9) < 0.01


def test_gamma_zero_is_rolling_conformal():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    pred = np.zeros(300)
    cal = np.abs(rng.normal(size=200))
    lo, hi, alphas = C.online_conformal(y, pred, pred, cal, 0.1, window=100, gamma=0.0)
    assert np.all(alphas == 0.1)
    scores = list(cal)
    for t in range(300):
        q = C.conformal_quantile(np.asarray(scores[-100:]), 0.9)
        assert abs(hi[t] - q) < 1e-12
        scores.append(abs(y[t]))


def test_aci_long_run_coverage_under_shift():
    # la garantie d'ACI vaut sans hypothese sur la loi : |erreur moyenne - alpha| <= (max(a1, 1 - a1) + gamma) / (gamma * T)
    rng = np.random.default_rng(2)
    T, alpha, gamma = 6000, 0.1, 0.005
    scale = np.where(np.arange(T) < 3000, 1.0, 4.0)
    y = rng.normal(size=T) * scale
    pred = np.zeros(T)
    cal = np.abs(rng.normal(size=300))
    lo, hi, _ = C.online_conformal(y, pred, pred, cal, alpha, window=200, gamma=gamma)
    miss = 1 - np.mean((lo <= y) & (y <= hi))
    bound = (max(alpha, 1 - alpha) + gamma) / (gamma * T)
    assert abs(miss - alpha) <= bound + 1e-12


def test_static_conformal_fails_under_shift_but_aci_recovers():
    rng = np.random.default_rng(3)
    T, alpha = 4000, 0.1
    scale = np.where(np.arange(T) < 1000, 1.0, 3.0)
    y = rng.normal(size=T) * scale
    pred = np.zeros(T)
    cal = np.abs(rng.normal(size=500))
    lo_s, hi_s = C.split_conformal(pred, pred, cal, alpha)
    lo_a, hi_a, _ = C.online_conformal(y, pred, pred, cal, alpha, window=200, gamma=0.01)
    after = slice(1500, T)
    cov_static = np.mean((lo_s <= y) & (y <= hi_s))
    cov_aci = np.mean(((lo_a <= y) & (y <= hi_a))[after])
    assert cov_static < 0.75
    assert abs(cov_aci - 0.9) < 0.03


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
