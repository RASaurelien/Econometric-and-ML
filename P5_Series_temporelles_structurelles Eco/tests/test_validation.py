import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

import series as sr
from series import kalman, structurel


def proches(a, b, tol=1e-8):
    return np.allclose(a, b, rtol=tol, atol=tol)


# VAR(2) trivarie simule
rng = np.random.default_rng(0)
T, k = 300, 3
A1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.3, 0.1], [0.0, 0.1, 0.4]])
Y = np.zeros((T, k))
for t in range(2, T):
    Y[t] = 0.2 + A1 @ Y[t - 1] + 0.1 * Y[t - 2] + rng.normal(size=k)
mine = sr.var(Y, 2)
ref = VAR(Y).fit(2, trend="c")


def test_var_contre_statsmodels():
    assert proches(mine.params, ref.params)
    assert proches(mine.sigma, ref.sigma_u)
    assert proches(mine.se, ref.stderr)
    assert mine.stable() == ref.is_stable()
    assert proches(mine.prevoir(6), ref.forecast(Y[-2:], 6))


def test_choix_de_l_ordre():
    ics, ordres = sr.choisir_ordre(Y, 6)
    ref_ordre = VAR(Y).select_order(6)
    for nom in ics:
        assert proches(ics[nom], np.array(ref_ordre.ics[nom][1:]))
        assert ordres[nom] == ref_ordre.selected_orders[nom]


@pytest.mark.parametrize("cause, effet", [([0], [1]), ([0, 1], [2])])
def test_granger(cause, effet):
    stat, p = mine.granger(cause, effet)
    r = ref.test_causality(effet if len(effet) > 1 else effet[0], cause if len(cause) > 1 else cause[0], kind="f")
    assert proches(stat, r.test_statistic) and proches(p, r.pvalue)


def test_irf_et_fevd_cholesky():
    assert proches(sr.irf_var(mine, 8), ref.irf(8).orth_irfs)
    assert proches(sr.fevd_var(mine, 10).transpose(1, 0, 2), ref.fevd(10).decomp)


# Blanchard-Quah : on retrouve la matrice d'impact vraie, et l'effet de long terme est triangulaire
def donnees_bq(n=40000, graine=1):
    r = np.random.default_rng(graine)
    A = np.array([[0.4, 0.2], [0.1, 0.3]])
    L = np.array([[1.0, 0.0], [0.5, 0.8]])
    B0 = (np.eye(2) - A) @ L
    Z = np.zeros((n, 2))
    e = r.normal(size=(n, 2))
    for t in range(1, n):
        Z[t] = A @ Z[t - 1] + B0 @ e[t]
    return Z, B0, L


def test_long_terme_retrouve_la_vraie_structure():
    Z, B0, L = donnees_bq()
    res = sr.var(Z, 1, constante=False)
    B = structurel.impact_long_terme(res.A, res.sigma)
    assert np.allclose(B, B0, atol=0.03)
    assert np.allclose(B @ B.T, res.sigma)
    longue = structurel.effet_long_terme(res.A) @ B
    assert abs(longue[0, 1]) < 1e-10 and np.allclose(longue, L, atol=0.03)


def test_bootstrap_irf_couvre_la_vraie_reponse():
    r = np.random.default_rng(7)
    A = np.array([[0.5, 0.2], [0.1, 0.4]])
    P = np.array([[1.0, 0.0], [0.4, 0.9]])
    vraie = structurel.irf(A[None], P, 4)
    couvert, essais = 0, 60
    for _ in range(essais):
        Z = np.zeros((200, 2))
        e = r.normal(size=(200, 2))
        for t in range(1, 200):
            Z[t] = A @ Z[t - 1] + P @ e[t]
        res = sr.var(Z, 1, constante=False)
        lo, hi = structurel.irf_bootstrap(res, 4, B=199, niveau=0.9, graine=int(r.integers(1e9)))
        couvert += np.mean((lo <= vraie) & (vraie <= hi))
    assert 0.8 < couvert / essais < 0.98


# Systeme cointegre : 3 variables, un vecteur de cointegration (1, -1, 0.5)
def donnees_coint(n=400, graine=0):
    r = np.random.default_rng(graine)
    beta, alpha = np.array([1.0, -1.0, 0.5]), np.array([-0.3, 0.1, 0.0])
    Z = np.zeros((n, 3))
    for t in range(1, n):
        Z[t] = Z[t - 1] + alpha * (Z[t - 1] @ beta) + r.normal(size=3)
    return Z


@pytest.mark.parametrize("det, ref_det", [("constante", 0), ("aucune", -1)])
def test_johansen_contre_statsmodels(det, ref_det):
    Z = donnees_coint()
    res = sr.johansen(Z, p=2, deterministe=det)
    r = coint_johansen(Z, det_order=ref_det, k_ar_diff=1)
    assert proches(res.valeurs_propres, r.eig)
    assert proches(res.trace, r.lr1)
    assert proches(res.maxeig, r.lr2)


@pytest.mark.parametrize("det, ref_det", [("constante", 0), ("aucune", -1)])
def test_valeurs_critiques_simulees_proches_des_tables(det, ref_det):
    table = coint_johansen(donnees_coint(), det_order=ref_det, k_ar_diff=1)
    for d in (1, 2, 3):
        cv = sr.valeurs_critiques(d, constante=det == "constante")
        assert abs(cv["trace"][0.95] - table.cvt[3 - d, 1]) < 0.4
        assert abs(cv["maxeig"][0.95] - table.cvm[3 - d, 1]) < 0.4


def test_rang_retenu():
    assert sr.johansen(donnees_coint(), p=2).rang() == 1


def test_vecm_contre_statsmodels():
    Z = donnees_coint()
    mine_v = sr.vecm(Z, p=2, r=1)
    r = VECM(Z, k_ar_diff=1, coint_rank=1, deterministic="co").fit()
    assert proches(mine_v.alpha.ravel(), r.alpha.ravel())
    assert proches(mine_v.beta.ravel(), r.beta.ravel())
    assert proches(mine_v.Gamma[0], r.gamma)
    assert proches(mine_v.mu, r.det_coef.ravel())
    assert proches(mine_v.sigma_ml, r.sigma_u)


# Le VECM ecrit en VAR de niveaux doit redonner les memes previsions : verification avec p = 3
def test_vecm_en_var_reproduit_la_dynamique():
    Z = donnees_coint()
    v = sr.vecm(Z, p=3, r=1)
    A, mu = v.en_var()
    t = 200
    dY = v.Pi @ Z[t - 1] + v.Gamma[0] @ (Z[t - 1] - Z[t - 2]) + v.Gamma[1] @ (Z[t - 2] - Z[t - 3]) + mu
    assert proches(Z[t - 1] + dY, mu + sum(A[j] @ Z[t - 1 - j] for j in range(3)))


# Filtre de Kalman : modele general avec observations manquantes
def modele_kalman(graine=0, n=200):
    r = np.random.default_rng(graine)
    q, m, s = 2, 3, 2
    T = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.2], [0.1, 0.0, 0.3]])
    mats = dict(Z=r.normal(size=(q, m)), H=np.diag([0.5, 0.8]), T=T, Q=0.7 * np.eye(s),
                R=r.normal(size=(m, s)), d=np.array([0.3, -0.2]), c=np.array([0.1, 0.0, 0.05]))
    y = r.normal(size=(n, q))
    y[r.random((n, q)) < 0.1] = np.nan
    return y, mats


def kalman_statsmodels(y, mats):
    ks = KalmanSmoother(2, 3, 2)
    ks.bind(y.T)
    for nom, cle in [("design", "Z"), ("obs_cov", "H"), ("transition", "T"), ("selection", "R"),
                     ("state_cov", "Q"), ("obs_intercept", "d"), ("state_intercept", "c")]:
        ks[nom] = mats[cle]
    ks.initialize_stationary()
    return ks.smooth()


@pytest.mark.skipif(not sr.cpp_disponible(), reason="bibliotheque C++ non compilee")
def test_kalman_cpp_contre_statsmodels():
    y, mats = modele_kalman()
    f = sr.filtre(y, **mats, moteur="cpp")
    r = kalman_statsmodels(y, mats)
    assert proches(f.loglik, r.llf, 1e-8)
    assert proches(f.a_filt, r.filtered_state.T, 1e-7)
    assert proches(f.a_pred, r.predicted_state.T, 1e-7)
    assert proches(f.P_filt, r.filtered_state_cov.transpose(2, 0, 1), 1e-7)


def test_lisseur_contre_statsmodels():
    y, mats = modele_kalman()
    f = sr.filtre(y, **mats)
    r = kalman_statsmodels(y, mats)
    a_s, P_s = sr.lisseur(f, mats["T"])
    assert proches(a_s, r.smoothed_state.T, 1e-7)
    assert proches(P_s, r.smoothed_state_cov.transpose(2, 0, 1), 1e-7)


@pytest.mark.skipif(not sr.cpp_disponible(), reason="bibliotheque C++ non compilee")
def test_cpp_et_numpy_identiques():
    y, mats = modele_kalman(graine=3, n=500)
    a = sr.filtre(y, **mats, moteur="cpp")
    b = sr.filtre(y, **mats, moteur="numpy")
    assert proches(a.loglik, b.loglik, 1e-10)
    assert proches(a.P_pred, b.P_pred, 1e-10)


def test_arma_contre_statsmodels():
    r = np.random.default_rng(5)
    e = r.normal(size=600)
    x = np.zeros(600)
    for t in range(1, 600):
        x[t] = 0.6 * x[t - 1] + e[t] + 0.4 * e[t - 1]
    x += 2
    mine_a = sr.arma(x, 1, 1)
    r_ref = ARIMA(x, order=(1, 0, 1), trend="c").fit()
    assert proches(mine_a.theta, r_ref.params, 1e-3)
    assert proches(mine_a.loglik, r_ref.llf, 1e-6)


def test_niveau_local_contre_statsmodels():
    r = np.random.default_rng(6)
    y = np.cumsum(0.5 * r.normal(size=300)) + r.normal(size=300)
    mine_l = sr.niveau_local(y)
    r_ref = UnobservedComponents(y, "llevel").fit(disp=0)
    assert proches(mine_l.theta, r_ref.params, 1e-3)
