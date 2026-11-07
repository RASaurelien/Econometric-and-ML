import numpy as np

from dgp import simulate_panel, split_cells
from fe_models import (pooled_ols, twfe, interactive_fe, twfe_cluster_se, ridge_effects,
                       add_group_means)
from embed_model import HAS_TORCH


def test_twfe_matches_dummy_regression_unbalanced():
    df = simulate_panel("additive", n_units=40, n_times=6, obs_prob=0.5, seed=1)
    beta, _ = twfe(df, 40, 6)
    D_u = np.eye(40)[df["unit"].values]
    D_t = np.eye(6)[df["time"].values][:, 1:]
    X = np.column_stack([df["x"].values, D_u, D_t])
    ref = np.linalg.lstsq(X, df["y"].values, rcond=None)[0][0]
    assert abs(beta - ref) < 1e-6


def test_twfe_cluster_estimate_matches_numpy():
    df = simulate_panel("additive", n_units=40, n_times=6, seed=2)
    beta, _ = twfe(df, 40, 6)
    beta_sm, se = twfe_cluster_se(df)
    assert abs(beta - beta_sm) < 1e-6
    assert se > 0


def test_additive_case_pooled_biased_twfe_ok():
    estimates = []
    pooled = []
    for s in range(10):
        df = simulate_panel("additive", n_units=400, n_times=10, seed=s)
        estimates.append(twfe(df, 400, 10)[0])
        pooled.append(pooled_ols(df)[0])
    assert abs(np.mean(estimates) - 1.0) < 0.03
    assert abs(np.mean(pooled) - 1.0) > 0.2


def test_interactive_case_twfe_biased_ife_ok():
    twfe_est, ife_est = [], []
    for s in range(6):
        df = simulate_panel("interactive", n_units=300, n_times=10, seed=s)
        twfe_est.append(twfe(df, 300, 10)[0])
        ife_est.append(interactive_fe(df, 300, 10, rank=2)[0])
    print("TWFE", np.mean(twfe_est), "IFE", np.mean(ife_est))
    assert abs(np.mean(twfe_est) - 1.0) > 0.2
    assert abs(np.mean(ife_est) - 1.0) < 0.08


def test_ridge_with_tiny_penalty_matches_twfe():
    df = simulate_panel("additive", n_units=40, n_times=6, obs_prob=0.7, seed=4)
    ref, _ = twfe(df, 40, 6)
    beta, _, _ = ridge_effects(df, 40, 6, lambdas=(1e-7,))
    assert abs(beta - ref) < 1e-4


def test_mundlak_reduces_shrinkage_bias():
    plain, mund = [], []
    for s in range(8):
        df = simulate_panel("additive", n_units=300, n_times=10, seed=s)
        train, test = split_cells(df, 0.2, seed=s)
        train, test = add_group_means(train, test, 300, 10)
        plain.append(ridge_effects(train, 300, 10, mundlak=False, seed=s)[0])
        mund.append(ridge_effects(train, 300, 10, mundlak=True, seed=s)[0])
    print("ridge", np.mean(plain), "ridge + Mundlak", np.mean(mund))
    assert abs(np.mean(mund) - 1.0) < abs(np.mean(plain) - 1.0)
    assert abs(np.mean(mund) - 1.0) < 0.05


def test_prediction_functions_return_one_value_per_row():
    df = simulate_panel("additive", n_units=50, n_times=6, seed=3)
    train, test = split_cells(df, 0.2, seed=0)
    for beta, pred in (pooled_ols(train), twfe(train, 50, 6), interactive_fe(train, 50, 6, rank=1)):
        assert len(pred(test)) == len(test)
        assert np.all(np.isfinite(pred(test)))


def test_embedding_interactive_recovers_beta():
    if not HAS_TORCH:
        return "skip"
    from embed_model import fit_embedding
    df = simulate_panel("interactive", n_units=300, n_times=10, seed=0)
    train, test = split_cells(df, 0.2, seed=0)
    train, test = add_group_means(train, test, 300, 10)
    beta, _, _ = fit_embedding(train, 300, 10, covariates=("x", "xbar_u", "xbar_t"), interactive=True)
    assert abs(beta - 1.0) < 0.2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            result = fn()
            print("ignore (torch absent)" if result == "skip" else "ok", name)
