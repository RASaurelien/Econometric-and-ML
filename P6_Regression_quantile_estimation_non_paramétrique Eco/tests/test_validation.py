import numpy as np
import pytest
import statsmodels.api as sm
from scipy.optimize import minimize
from statsmodels.datasets import engel
from statsmodels.nonparametric import bandwidths
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_regression import KernelReg

import quantnp as qn
from quantnp.noyaux import NOYAUX

engel_data = engel.load_pandas().data
Y_E, X_E = engel_data.foodexp.values, engel_data.income.values


def proches(a, b, tol=1e-8):
    return np.allclose(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("tau", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_quantile_contre_statsmodels(tau):
    mine = qn.regression_quantile(Y_E, X_E, tau, vcov="iid")
    ref = sm.QuantReg(Y_E, sm.add_constant(X_E)).fit(q=tau, vcov="iid")
    assert proches(mine.beta, ref.params, 1e-4)
    assert proches(mine.se, ref.bse, 1e-4)
    assert proches(mine.pseudo_r2, ref.prsquared, 1e-6)
    robuste = qn.regression_quantile(Y_E, X_E, tau)
    ref_r = sm.QuantReg(Y_E, sm.add_constant(X_E)).fit(q=tau, vcov="robust")
    assert np.allclose(robuste.se, ref_r.bse, rtol=0.03)


# Le programme lineaire doit etre au moins aussi bon que statsmodels et que toute perturbation
def test_optimalite():
    X = np.column_stack([np.ones(len(X_E)), X_E])
    res = qn.regression_quantile(Y_E, X_E, 0.3)
    ref = sm.QuantReg(Y_E, X).fit(q=0.3)
    assert res.objectif <= qn.perte(Y_E - X @ ref.params, 0.3) + 1e-9
    r = np.random.default_rng(0)
    for _ in range(200):
        b = res.beta + r.normal(size=2) * [5, 0.05]
        assert qn.perte(Y_E - X @ b, 0.3) >= res.objectif - 1e-9


def test_mediane_egale_moindres_ecarts_absolus():
    r = np.random.default_rng(1)
    x = r.normal(size=300)
    y = 1 + 2 * x + r.standard_t(3, size=300)
    X = np.column_stack([np.ones(300), x])
    res = qn.regression_quantile(y, x, 0.5)
    lad = minimize(lambda b: np.abs(y - X @ b).sum(), [1, 2], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12})
    assert res.objectif <= lad.fun + 1e-9


# Modele a echelle et position : y = 1 + 2x + (1 + 0.5x) u, donc beta_tau = (1 + q_tau, 2 + 0.5 q_tau)
def test_quantiles_retrouvent_la_vraie_structure():
    r = np.random.default_rng(2)
    n = 20000
    x = r.uniform(0, 2, n)
    y = 1 + 2 * x + (1 + 0.5 * x) * r.normal(size=n)
    from scipy.stats import norm
    for tau in (0.1, 0.5, 0.9):
        q = norm.ppf(tau)
        res = qn.regression_quantile(y, x, tau)
        assert np.allclose(res.beta, [1 + q, 2 + 0.5 * q], atol=0.06)


def test_reordonnancement_supprime_les_croisements():
    r = np.random.default_rng(3)
    x = r.uniform(0, 1, 120)
    y = 1 + x + (0.2 + 2 * x) * r.normal(size=120)
    res = qn.regressions_quantiles(y, x, (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95))
    grille = np.column_stack([np.ones(50), np.linspace(-0.5, 1.5, 50)])
    brut = qn.predire_quantiles(res, grille, rearranger=False)
    trie = qn.predire_quantiles(res, grille)
    assert np.any(np.diff(brut, axis=1) < 0)
    assert np.all(np.diff(trie, axis=1) >= 0)


def test_bootstrap_proche_de_la_variance_robuste():
    r = np.random.default_rng(4)
    x = r.normal(size=500)
    y = 1 + x + r.normal(size=500)
    a = qn.regression_quantile(y, x, 0.5, vcov="robuste")
    b = qn.regression_quantile(y, x, 0.5, vcov="bootstrap", B=300, graine=0)
    assert np.allclose(a.se, b.se, rtol=0.3)


def test_egalite_des_pentes():
    r = np.random.default_rng(5)
    x = r.uniform(0, 2, 800)
    homo = 1 + x + r.normal(size=800)
    hetero = 1 + x + (0.3 + x) * r.normal(size=800)
    assert qn.egalite_pentes(homo, x, (0.1, 0.9), B=150, graine=0)[1] > 0.05
    assert qn.egalite_pentes(hetero, x, (0.1, 0.9), B=150, graine=0)[1] < 0.01


# Partie non parametrique
r = np.random.default_rng(0)
N_NP = 300
X_NP = np.sort(r.uniform(0, 3, N_NP))
Y_NP = np.sin(2 * X_NP) + 0.3 * r.normal(size=N_NP)
GRILLE = np.linspace(0.2, 2.8, 15)


def test_fenetres_de_reference():
    assert proches(qn.bw_silverman(X_NP), bandwidths.bw_silverman(X_NP))
    assert proches(qn.bw_scott(X_NP), bandwidths.bw_scott(X_NP))


def test_densite_contre_statsmodels():
    kde = KDEUnivariate(X_NP)
    kde.fit(kernel="gau", bw=0.25, fft=False)
    assert proches(qn.densite(X_NP, GRILLE, 0.25), kde.evaluate(GRILLE))


@pytest.mark.parametrize("noyau", list(NOYAUX))
def test_densite_positive_et_integre_a_un(noyau):
    g = np.linspace(-3, 6, 4000)
    f = qn.densite(X_NP, g, 0.3, noyau)
    assert np.all(f >= 0)
    assert abs(np.sum(f) * (g[1] - g[0]) - 1) < 1e-3


@pytest.mark.parametrize("degre, reg_type", [(0, "lc"), (1, "ll")])
def test_regression_locale_contre_statsmodels(degre, reg_type):
    ref = KernelReg(Y_NP, X_NP, var_type="c", reg_type=reg_type, bw=[0.25]).fit(GRILLE)[0]
    assert proches(qn.lisser(X_NP, Y_NP, GRILLE, 0.25, "gaussien", degre), ref, 1e-7)


@pytest.mark.parametrize("noyau", ["gaussien", "epanechnikov"])
def test_lineaire_local_reproduit_les_droites(noyau):
    L = qn.poids_locaux(X_NP, GRILLE, 0.4, noyau, 1)
    assert proches(L.sum(axis=1), 1.0, 1e-10)
    assert proches(L @ X_NP, GRILLE, 1e-10)
    assert proches(qn.lisser(X_NP, 2 - 3 * X_NP, GRILLE, 0.4, noyau, 1), 2 - 3 * GRILLE, 1e-9)


def loo_brut(x, y, h, degre):
    erreurs = []
    for i in range(len(x)):
        m = np.ones(len(x), bool)
        m[i] = False
        erreurs.append(y[i] - qn.lisser(x[m], y[m], x[i:i + 1], h, "gaussien", degre)[0])
    return np.mean(np.array(erreurs) ** 2)


@pytest.mark.parametrize("degre", [0, 1])
def test_validation_croisee_sans_refaire_les_estimations(degre):
    xs, ys = X_NP[:80], Y_NP[:80]
    assert proches(qn.cv_loo(xs, ys, 0.3, "gaussien", degre), loo_brut(xs, ys, 0.3, degre), 1e-10)


@pytest.mark.parametrize("degre, reg_type", [(0, "lc"), (1, "ll")])
def test_fenetre_cv_contre_statsmodels(degre, reg_type):
    h = qn.choisir_h(X_NP, Y_NP, degre=degre)
    ref = KernelReg(Y_NP, X_NP, var_type="c", reg_type=reg_type, bw="cv_ls").bw[0]
    assert abs(h / ref - 1) < 0.01
    assert qn.cv_loo(X_NP, Y_NP, h, "gaussien", degre) <= qn.cv_loo(X_NP, Y_NP, 1.3 * h, "gaussien", degre)
    assert qn.cv_loo(X_NP, Y_NP, h, "gaussien", degre) <= qn.cv_loo(X_NP, Y_NP, 0.7 * h, "gaussien", degre)


# Couverture des intervalles ponctuels en un point, fonction lisse et fenetre fixe
def test_couverture_des_intervalles():
    rg = np.random.default_rng(8)
    couvert, essais = 0, 150
    for _ in range(essais):
        x = rg.uniform(0, 3, 400)
        y = np.sin(2 * x) + (0.2 + 0.2 * x) * rg.normal(size=400)
        _, bas, haut = qn.intervalle(x, y, np.array([1.5]), 0.2)
        couvert += bas[0] <= np.sin(3.0) <= haut[0]
    assert 0.88 < couvert / essais < 0.99


# Modele partiellement lineaire : x est correle a z, donc les MCO sans g(z) sont biaises
def test_robinson_contre_mco_naif():
    rg = np.random.default_rng(9)
    n, essais = 500, 120
    beta, ses, ols_b, couvert = [], [], [], 0
    for _ in range(essais):
        z = rg.uniform(0, 1, n)
        x = z + 0.5 * rg.normal(size=n)
        y = 1.5 * x + np.sin(2 * np.pi * z) + 0.5 * rg.normal(size=n)
        res = qn.robinson(y, x, z)
        beta.append(res.beta[0])
        ses.append(res.se[0])
        couvert += abs(res.beta[0] - 1.5) < 1.96 * res.se[0]
        ols_b.append(sm.OLS(y, sm.add_constant(x)).fit().params[1])
    assert abs(np.mean(beta) - 1.5) < 0.03
    assert abs(np.mean(ols_b) - 1.5) > 0.1
    assert 0.88 < couvert / essais < 0.99
    assert 0.7 < np.std(beta) / np.mean(ses) < 1.3


def test_robinson_retrouve_la_partie_non_parametrique():
    rg = np.random.default_rng(10)
    n = 3000
    z = rg.uniform(0, 1, n)
    x = z + 0.5 * rg.normal(size=n)
    y = 1.5 * x + np.sin(2 * np.pi * z) + 0.5 * rg.normal(size=n)
    res = qn.robinson(y, x, z)
    g = np.linspace(0.1, 0.9, 9)
    assert np.allclose(res.g(g) - res.g(g).mean(), np.sin(2 * np.pi * g) - np.sin(2 * np.pi * g).mean(), atol=0.15)
