import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.iv import IV2SLS, IVGMM

from econometrie import mco, mcg, mcgf, mc2e, gmm, wald, breusch_pagan, j_hansen

# Donnees simulees avec erreurs heteroscedastiques
rng = np.random.default_rng(42)
n = 500
X = rng.normal(size=(n, 3))
y = 1 + X @ [0.5, -1, 2] + rng.normal(size=n) * (1 + np.abs(X[:, 0]))
XC = sm.add_constant(X)

# Donnees avec un regresseur endogene et deux instruments
z = rng.normal(size=(n, 2))
w = rng.normal(size=n)
u = rng.normal(size=n)
x_endo = z @ [1, 0.5] + 0.8 * u + rng.normal(size=n)
y_iv = 1 + 2 * x_endo + 0.5 * w + u * (1 + 0.5 * np.abs(w))


def test_mco_classique():
    mine = mco(y, X)
    ref = sm.OLS(y, XC).fit()
    assert np.allclose(mine.beta, ref.params)
    assert np.allclose(mine.se, ref.bse)
    assert np.allclose(mine.p, ref.pvalues)
    assert np.isclose(mine.r2(), ref.rsquared)


@pytest.mark.parametrize("h", ["HC0", "HC1", "HC2", "HC3"])
def test_mco_hc(h):
    ref = sm.OLS(y, XC).fit(cov_type=h)
    assert np.allclose(mco(y, X, cov=h).se, ref.bse)


def test_mco_hac():
    ref = sm.OLS(y, XC).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    assert np.allclose(mco(y, X, cov="HAC", lags=4).se, ref.bse)


def test_mco_cluster():
    g = rng.integers(0, 25, n)
    ref = sm.OLS(y, XC).fit(cov_type="cluster", cov_kwds={"groups": g})
    assert np.allclose(mco(y, X, cov="cluster", groupes=g).se, ref.bse)


def test_wald():
    R = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    stat, p = wald(mco(y, X), R)
    ref = sm.OLS(y, XC).fit().wald_test(R, use_f=False, scalar=True)
    assert np.isclose(stat, float(np.squeeze(ref.statistic)))
    assert np.isclose(p, float(np.squeeze(ref.pvalue)))


def test_breusch_pagan():
    res = mco(y, X)
    ref = het_breuschpagan(res.residus, res.X)
    stat, p = breusch_pagan(res)
    assert np.isclose(stat, ref[0])
    assert np.isclose(p, ref[1])


def test_mcg():
    variances = (1 + np.abs(X[:, 0])) ** 2
    ref = sm.WLS(y, XC, weights=1 / variances).fit()
    mine = mcg(y, X, variances)
    assert np.allclose(mine.beta, ref.params)
    assert np.allclose(mine.se, ref.bse)


# Variance de la forme exp(a + b x), celle que le MCGF sait modeliser
def test_mcgf_proche_de_la_vraie_valeur():
    r = np.random.default_rng(7)
    m = 4000
    Xf = r.normal(size=(m, 3))
    yf = 1 + Xf @ [0.5, -1, 2] + r.normal(size=m) * np.exp(0.8 * Xf[:, 0])
    assert np.allclose(mcgf(yf, Xf).beta, [1, 0.5, -1, 2], atol=0.1)


def test_mc2e():
    mine = mc2e(y_iv, np.column_stack([w, x_endo]), np.column_stack([w, z]))
    ref = IV2SLS(y_iv, sm.add_constant(w), x_endo, z).fit(
        cov_type="unadjusted", debiased=True
    )
    assert np.allclose(mine.beta, ref.params.values)
    assert np.allclose(mine.se, ref.std_errors.values)


def test_gmm_et_j():
    Xg = np.column_stack([w, x_endo])
    Zg = np.column_stack([w, z])
    mine = gmm(y_iv, Xg, Zg)
    ref = IVGMM(y_iv, sm.add_constant(w), x_endo, z, weight_type="robust").fit(
        iter_limit=2
    )
    assert np.allclose(mine.beta, ref.params.values, atol=1e-6)
    assert np.allclose(mine.se, ref.std_errors.values, atol=1e-6)
    # linearmodels garde les poids de l'etape 1 pour le J, d'ou un petit ecart
    assert np.isclose(j_hansen(mine)[0], ref.j_stat.stat, rtol=1e-2)
