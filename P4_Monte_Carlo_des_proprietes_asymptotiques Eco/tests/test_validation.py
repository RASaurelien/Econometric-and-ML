import numpy as np
import pytest
import statsmodels.api as sm
from linearmodels.iv import IV2SLS, IVLIML

import montecarlo as mc
from montecarlo import bootstrap as bs
from montecarlo import simulation as sim


def proches(a, b, tol=1e-8):
    return np.allclose(a, b, rtol=tol, atol=tol)


# Quelques echantillons avec heteroscedasticite et forts leviers
rng = np.random.default_rng(0)
X, y = sim.generer(4, 60, rng, hetero=0.4, regresseurs="asymetrique")


@pytest.mark.parametrize("variante", ["classique", "HC0", "HC1", "HC2", "HC3"])
def test_mco_batch_contre_statsmodels(variante):
    mine = mc.MCOBatch(y, X)
    for r in range(4):
        ref = sm.OLS(y[r], X[r]).fit(cov_type="nonrobust" if variante == "classique" else variante)
        assert proches(mine.beta[r], ref.params)
        assert proches(mine.se(variante)[r], ref.bse)


def test_iv_contre_linearmodels():
    r = np.random.default_rng(1)
    yy, x, z = sim.generer_iv(3, 200, 5, 10, r)
    iv = mc.IVBatch(yy, x, z)
    b2, s2 = iv.deux_smc()
    bl, sl = iv.liml()
    for i in range(3):
        ref2 = IV2SLS(yy[i], None, x[i], z[i]).fit(cov_type="unadjusted")
        refl = IVLIML(yy[i], None, x[i], z[i]).fit(cov_type="unadjusted")
        assert proches(b2[i], ref2.params.values) and proches(s2[i], ref2.std_errors.values)
        assert proches(bl[i], refl.params.values) and proches(sl[i], refl.std_errors.values)


# Boucle naive : on refait chaque regression avec statsmodels, tirage par tirage
def test_wild_bootstrap_contre_boucle():
    r = np.random.default_rng(2)
    B = 15
    w = r.choice([-1.0, 1.0], size=(4, B, 60))
    t_obs, t_star = bs.wild_t(X, y, 1, 0.5, w, "HC1")
    for c in range(4):
        xj = X[c][:, 1]
        restreint = sm.OLS(y[c] - 0.5 * xj, X[c][:, :1]).fit()
        e_r = restreint.resid
        ajuste = restreint.fittedvalues + 0.5 * xj
        for b in range(B):
            ys = ajuste + e_r * w[c, b]
            ref = sm.OLS(ys, X[c]).fit(cov_type="HC1")
            assert proches(t_star[c, b], (ref.params[1] - 0.5) / ref.bse[1])
        ref = sm.OLS(y[c], X[c]).fit(cov_type="HC1")
        assert proches(t_obs[c], (ref.params[1] - 0.5) / ref.bse[1])


def test_bootstrap_paires_contre_boucle():
    r = np.random.default_rng(3)
    B = 15
    idx = r.integers(0, 60, size=(4, B, 60))
    t_obs, t_star = bs.pairs_t(X, y, 1, 0.5, idx, "HC1")
    for c in range(4):
        b_hat = sm.OLS(y[c], X[c]).fit().params[1]
        for b in range(B):
            ref = sm.OLS(y[c][idx[c, b]], X[c][idx[c, b]]).fit(cov_type="HC1")
            assert proches(t_star[c, b], (ref.params[1] - b_hat) / ref.bse[1])


# Verification des resultats attendus par la theorie, avec une marge pour le bruit de simulation
def test_taille_homoscedastique_proche_de_5_pour_cent():
    res = mc.taille(n=100, R=3000, methodes=["classique", "HC3"], graine=1)
    assert 0.04 < res["classique"] < 0.06
    assert 0.04 < res["HC3"] < 0.065


def test_heteroscedasticite_fausse_le_test_classique():
    res = mc.taille(n=50, hetero=0.5, regresseurs="asymetrique", R=2000, B=99, graine=2)
    assert res["classique"] > 0.2
    assert res["HC0"] > res["HC2"] > res["HC3"]
    assert res["HC3"] < 0.10 and res["wild"] < 0.09 and res["pairs"] < 0.08


def test_puissance_croissante_et_egale_a_la_taille_sous_h0():
    entetes, lignes = mc.puissance(n=50, effets=(0.0, 0.2, 0.4), R=1500, B=99, graine=3)
    classique = [ligne[1] for ligne in lignes]
    hc3 = [ligne[2] for ligne in lignes]
    assert 0.03 < hc3[0] < 0.08
    assert classique[0] < classique[1] < classique[2]
    assert hc3[0] < hc3[1] < hc3[2]


def test_biais_et_consistance():
    _, lignes = mc.biais_consistance(ns=(400, 1600), R=3000, graine=4)
    (n1, exo1, sd1, endo1, iv1, sdiv1), (n2, exo2, sd2, endo2, iv2, sdiv2) = lignes
    assert abs(endo2 - 0.6 / 1.48) < 0.02
    assert abs(exo2) < 0.01 and abs(iv2) < 0.02
    assert abs(sdiv1 / sdiv2 - 1) < 0.1


def test_instruments_faibles():
    _, lignes = mc.instruments_faibles(mu2s=(1, 30), R=4000, graine=5)
    faible, fort = lignes
    assert faible[3] > 0.5 and abs(faible[3] - faible[2]) < 0.2 * faible[2] + 0.2
    assert abs(fort[4]) < 0.02
    assert faible[5] > 0.3
    assert 0.035 < faible[7] < 0.065 and 0.035 < fort[7] < 0.065
