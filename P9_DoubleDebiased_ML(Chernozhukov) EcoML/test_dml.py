import numpy as np

import dml_lib as dl
from simulate import dgp_plr, dgp_irm, irm_nuisance, plr_nuisance, draw_x


def test_irm_truth_is_ate():
    rng = np.random.default_rng(0)
    X = draw_x(200000, 10, rng)
    _, g0, g1 = irm_nuisance(X, 0.5, "complex")
    assert abs(np.mean(g1 - g0) - 0.5) < 0.01


def test_plr_residualisation_with_true_nuisances():
    # avec les vraies fonctions de nuisance, l'estimateur doit etre sans biais
    rng = np.random.default_rng(1)
    est = []
    for _ in range(200):
        Y, D, X = dgp_plr(1000, 10, 0.5, rng, "complex")
        m, g = plr_nuisance(X, "complex")
        res_d = D - m
        res_y = Y - (0.5 * m + g)
        est.append((res_d @ res_y) / (res_d @ res_d))
    est = np.array(est)
    assert abs(est.mean() - 0.5) < 3 * est.std() / np.sqrt(len(est))


def check_estimator(draw, fn, theta, reps, tol_cover=0.85):
    est, cover = [], []
    for r in range(reps):
        Y, D, X = draw(np.random.default_rng(100 + r))
        t, se = fn(Y, D, X)
        est.append(t)
        cover.append(abs(t - theta) <= 1.96 * se)
    est = np.array(est)
    mc_se = est.std() / np.sqrt(reps)
    return est.mean() - theta, mc_se, np.mean(cover)


def test_dml_plr_lasso_linear_design():
    # lasso bien specifie : le DML doit etre sans biais et bien calibre
    draw = lambda rng: dgp_plr(800, 50, 0.5, rng, "linear")
    bias, mc_se, cover = check_estimator(draw, lambda Y, D, X: dl.dml_plr(Y, D, X, "lasso"), 0.5, 40)
    print("PLR lasso lineaire : biais %.3f (mc se %.3f), couverture %.2f" % (bias, mc_se, cover))
    assert abs(bias) < 3 * mc_se
    assert cover > 0.85


def test_dml_irm_lasso_linear_design():
    draw = lambda rng: dgp_irm(1000, 50, 0.5, rng, "linear")
    bias, mc_se, cover = check_estimator(draw, lambda Y, D, X: dl.dml_irm(Y, D, X, "lasso"), 0.5, 40)
    print("IRM lasso lineaire : biais %.3f (mc se %.3f), couverture %.2f" % (bias, mc_se, cover))
    assert abs(bias) < 3 * mc_se
    assert cover > 0.85


def test_naive_ols_is_biased():
    draw = lambda rng: dgp_plr(800, 50, 0.5, rng, "linear")
    bias, mc_se, _ = check_estimator(draw, dl.ols_naive, 0.5, 30)
    assert abs(bias) > 5 * mc_se


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
