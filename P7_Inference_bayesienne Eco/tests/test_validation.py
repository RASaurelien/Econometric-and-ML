import math

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit

import bayes as by
from bayes import coeurs, reference
from bayes.diagnostics import ess, rhat
from bayes.rng import Rng

trapz = getattr(np, "trapezoid", None) or np.trapz
COMPILES = [b for b in ("cpp", "rust") if coeurs.LIBS[b] is not None]
sans_cpp = pytest.mark.skipif(not COMPILES, reason="aucun cœur compile (make)")
sur_tous = pytest.mark.parametrize("backend", COMPILES)


def proches(a, b, tol=1e-8):
    return np.allclose(a, b, rtol=tol, atol=tol)


# Petit panel pour comparer les implementations
def donnees_panel(N=30, T=4, k=3, graine=0):
    r = np.random.default_rng(graine)
    grp = np.repeat(np.arange(N), T)
    X = np.column_stack([np.ones(N * T), r.normal(size=(N * T, k - 1))])
    y = X @ np.arange(1, k + 1) + 0.8 * r.normal(size=N)[grp] + 0.7 * r.normal(size=N * T)
    hyp = np.array([1e4, 0.01, 0.01, 1.0, 1.0])
    init = np.concatenate([np.linalg.lstsq(X, y, rcond=None)[0], np.zeros(N), [0.5, 1.0]])
    return y, X, grp, N, hyp, init


# Le generateur : memes flux de uniformes, normales et gamma dans les trois langages
@sur_tous
@pytest.mark.parametrize("graine", [0, 1, 123456789, 2 ** 63 + 5])
def test_generateur_identique(backend, graine):
    assert np.array_equal(coeurs.flux(backend, graine, 300), coeurs.flux("python", graine, 300))


@sur_tous
def test_generateur_statistiquement_correct(backend):
    n = 200000
    f = coeurs.flux(backend, 3, n).reshape(4, n)
    unif, norm_, g25, g04 = f
    assert abs(unif.mean() - 0.5) < 4 * math.sqrt(1 / 12 / n) and abs(unif.var() - 1 / 12) < 0.002
    assert abs(norm_.mean()) < 4 / math.sqrt(n) and abs(norm_.var() - 1) < 0.02
    assert abs(stats.skew(norm_)) < 0.03 and abs(stats.kurtosis(norm_)) < 0.06
    for g, a in ((g25, 2.5), (g04, 0.4)):
        assert abs(g.mean() - a) < 4 * math.sqrt(a / n) and abs(g.var() - a) < 0.05 * a
    assert stats.kstest(unif, "uniform").pvalue > 1e-3
    assert stats.kstest(g25, "gamma", args=(2.5,)).pvalue > 1e-3


# Chaines de Gibbs et de Metropolis-Hastings identiques tirage par tirage
@sur_tous
def test_gibbs_identique_a_la_reference(backend):
    y, X, grp, N, hyp, init = donnees_panel()
    ref = coeurs.gibbs("python", y, X, grp, N, init, hyp, 400, 100, 2, 7)
    res = coeurs.gibbs(backend, y, X, grp, N, init, hyp, 400, 100, 2, 7)
    assert ref[0].shape == res[0].shape == (150, 5)
    assert proches(res[0], ref[0], 1e-10) and proches(res[1], ref[1], 1e-10)


@sur_tous
def test_mh_identique_a_la_reference(backend):
    y, X, grp, N, _, _ = donnees_panel(N=40, k=3)
    yb = (np.random.default_rng(1).random(len(y)) < expit(X @ [0.5, 1, -1])).astype(float)
    L = 0.3 * np.eye(3)
    ref = coeurs.mh("python", yb, X, 10.0, L, np.zeros(3), 3000, 500, 3, 11)
    res = coeurs.mh(backend, yb, X, 10.0, L, np.zeros(3), 3000, 500, 3, 11)
    assert res[1] == ref[1] and 0 < res[1] < 3000
    assert np.array_equal(res[0], ref[0])


@sur_tous
def test_reproductibilite(backend):
    y, X, grp, N, hyp, init = donnees_panel()
    a = coeurs.gibbs(backend, y, X, grp, N, init, hyp, 300, 100, 1, 5)[0]
    b = coeurs.gibbs(backend, y, X, grp, N, init, hyp, 300, 100, 1, 5)[0]
    c = coeurs.gibbs(backend, y, X, grp, N, init, hyp, 300, 100, 1, 6)[0]
    assert np.array_equal(a, b) and not np.allclose(a, c)


# Chaque loi conditionnelle de Gibbs contre sa formule exacte : apres un balayage, on standardise
# chaque quantite tiree par la loi conditionnelle correspondante, le resultat doit etre N(0, 1)
# (ou Gamma pour les variances).
def test_lois_conditionnelles_exactes():
    y, X, grp, N, hyp, init = donnees_panel(N=8, T=3, k=2)
    k, n = X.shape[1], len(y)
    var_beta, a_s, b_s, a_t, b_t = hyp
    XtX, Tg = X.T @ X, np.bincount(grp).astype(float)
    rng = Rng(21)
    etat = dict(beta=init[:k].copy(), alpha=np.zeros(N), sigma2=0.7, tau2=0.9)
    za, zb, ga, gt = [], [], [], []
    for _ in range(6000):
        avant = dict(etat)
        reference.balayage(rng, y, X, grp, N, etat, hyp, XtX, Tg)
        S = np.bincount(grp, weights=y - X @ avant["beta"], minlength=N)
        prec = Tg / avant["sigma2"] + 1 / avant["tau2"]
        za.append((etat["alpha"] - S / avant["sigma2"] / prec) * np.sqrt(prec))
        P = XtX / avant["sigma2"] + np.eye(k) / var_beta
        V = np.linalg.inv(P)
        m = V @ (X.T @ (y - etat["alpha"][grp]) / avant["sigma2"])
        zb.append(np.linalg.solve(np.linalg.cholesky(V), etat["beta"] - m))
        r = y - X @ etat["beta"] - etat["alpha"][grp]
        ga.append((b_s + 0.5 * r @ r) / etat["sigma2"])
        gt.append((b_t + 0.5 * etat["alpha"] @ etat["alpha"]) / etat["tau2"])
    za, zb = np.array(za).ravel(), np.array(zb).ravel()
    assert abs(za.mean()) < 0.03 and abs(za.var() - 1) < 0.04
    assert abs(zb.mean()) < 0.04 and abs(zb.var() - 1) < 0.05
    for g, forme in ((ga, a_s + n / 2), (gt, a_t + N / 2)):
        g = np.array(g)
        assert abs(g.mean() - forme) < 4 * math.sqrt(forme / len(g))
        assert abs(g.var() / forme - 1) < 0.08


# Test de Geweke (2004) : en alternant tirage des donnees sachant les parametres et un balayage de
# Gibbs, la loi des parametres doit rester la loi a priori. On compare les moments obtenus aux
# moments exacts du prior.
def test_geweke():
    N, T, k = 4, 3, 2
    var_beta, a_s, b_s, a_t, b_t = 4.0, 3.0, 2.0, 3.0, 2.0
    hyp = [var_beta, a_s, b_s, a_t, b_t]
    r = np.random.default_rng(0)
    grp = np.repeat(np.arange(N), T)
    X = r.normal(size=(N * T, k))
    XtX, Tg = X.T @ X, np.bincount(grp).astype(float)
    rng = Rng(5)
    etat = dict(beta=r.normal(size=k) * 2, alpha=r.normal(size=N), sigma2=1.0, tau2=1.0)
    n_pas = 60000
    suivi = np.empty((n_pas, 5))
    for t in range(n_pas):
        y = X @ etat["beta"] + etat["alpha"][grp] + math.sqrt(etat["sigma2"]) * r.normal(size=N * T)
        reference.balayage(rng, y, X, grp, N, etat, hyp, XtX, Tg)
        suivi[t] = [etat["beta"][0], etat["beta"][0] ** 2, 1 / etat["sigma2"], etat["tau2"], etat["alpha"][0] ** 2]
    attendu = [0.0, var_beta, a_s / b_s, b_t / (a_t - 1), b_t / (a_t - 1)]
    lots = suivi[: n_pas // 40 * 40].reshape(40, -1, 5).mean(axis=1)
    for j in range(5):
        se = lots[:, j].std(ddof=1) / math.sqrt(40)
        assert abs(suivi[:, j].mean() - attendu[j]) < 4.5 * se, j


# Posterieur du panel contre le maximum de vraisemblance restreint de MixedLM
def test_gibbs_contre_mixedlm():
    r = np.random.default_rng(1)
    N, T = 100, 8
    ids = np.repeat(np.arange(N), T)
    X = r.normal(size=(N * T, 2))
    effets = 0.8 * r.normal(size=N)
    y = 1 + X @ [2, -1] + effets[ids] + r.normal(size=N * T)
    res = by.gibbs_panel(y, X, ids, n_iter=3000, burn=1000, graine=2)
    mm = sm.MixedLM(y, sm.add_constant(X), ids).fit(reml=True)
    moy = res.moyennes()
    assert np.allclose(moy[:3], mm.fe_params, atol=0.02)
    assert abs(moy[3] / mm.scale - 1) < 0.05 and abs(moy[4] / np.asarray(mm.cov_re)[0, 0] - 1) < 0.12
    lignes = res.resume().lignes
    assert all(l[5] < 1.01 for l in lignes) and all(l[4] > 300 for l in lignes)
    assert np.corrcoef(res.alpha_moyen, effets)[0, 1] > 0.8


# Logit bayesien. Cas a un parametre : le posterieur se calcule exactement sur une grille.
def test_mh_contre_integration_numerique():
    r = np.random.default_rng(3)
    x = r.normal(size=200)
    y = (r.random(200) < expit(0.8 * x)).astype(float)
    sd = 3.0
    res = by.mh_logit(y, x, n_iter=30000, burn=3000, graine=4, prior_sd=sd, constante=False)
    grille = np.linspace(-1, 3, 8001)
    logp = np.array([np.sum(y * g * x - np.logaddexp(0, g * x)) for g in grille]) - 0.5 * grille ** 2 / sd ** 2
    dens = np.exp(logp - logp.max())
    dens /= trapz(dens, grille)
    moy = trapz(grille * dens, grille)
    ecart = math.sqrt(trapz((grille - moy) ** 2 * dens, grille))
    t = res.tirages[:, :, 0]
    se = t.std() / math.sqrt(ess(t))
    assert abs(t.mean() - moy) < 4 * se
    assert abs(t.std() / ecart - 1) < 0.03
    cdf = np.cumsum(dens) * (grille[1] - grille[0])
    for q in (0.025, 0.5, 0.975):
        assert abs(np.quantile(t, q) - np.interp(q, cdf, grille)) < 0.05 * ecart * 2


def test_mh_contre_maximum_de_vraisemblance():
    r = np.random.default_rng(4)
    n = 1000
    X = r.normal(size=(n, 3))
    y = (r.random(n) < expit(0.5 + X @ [1, -0.5, 0.3])).astype(float)
    res = by.mh_logit(y, X, n_iter=12000, burn=2000, graine=5, prior_sd=100.0)
    ref = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    moy = res.moyennes()
    sd = res.tirages.reshape(-1, 4).std(axis=0)
    assert np.all(np.abs(moy - ref.params) < 0.2 * ref.bse)
    assert np.all(np.abs(sd / ref.bse - 1) < 0.06)
    assert all(0.15 < a < 0.5 for a in res.acceptation)
    assert all(l[5] < 1.01 for l in res.resume().lignes)


# Diagnostics : l'ESS d'un AR(1) vaut n (1 - rho) / (1 + rho), le R-hat detecte les chaines non melangees
def test_diagnostics():
    r = np.random.default_rng(6)
    rho, n, m = 0.9, 40000, 4
    x = np.zeros((m, n))
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + math.sqrt(1 - rho ** 2) * r.normal(size=m)
    assert abs(ess(x) / (m * n * (1 - rho) / (1 + rho)) - 1) < 0.15
    iid = r.normal(size=(4, 5000))
    assert rhat(iid) < 1.01 and abs(ess(iid) / iid.size - 1) < 0.15
    assert rhat(iid + np.arange(4)[:, None] * 2) > 1.5


def test_interface_de_haut_niveau():
    y, X, grp, N, _, _ = donnees_panel(N=20, T=5, k=3)
    res = by.gibbs_panel(y, X[:, 1:], grp, n_iter=400, burn=100, chaines=3, graine=1, backend="python",
                         noms=["a", "b"])
    assert res.tirages.shape == (3, 300, 5) and res.noms == ["const", "a", "b", "sigma2", "tau2"]
    assert res.alpha_moyen.shape == (N,) and "R-hat" in str(res.resume())
    with pytest.raises(ValueError):
        by.gibbs_panel(y, X[:, 1:], grp, n_iter=100, burn=100, backend="python")
