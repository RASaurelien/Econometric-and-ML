import numpy as np
import pytest
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.base.model import GenericLikelihoodModel

import mle
from mle.optimiseurs import newton, bfgs

# Donnees communes : un regresseur continu et une variable binaire
rng = np.random.default_rng(0)
n = 2000
X = np.column_stack([rng.normal(size=n), rng.integers(0, 2, n)])
XC = sm.add_constant(X)
xb = 0.5 + 1.0 * X[:, 0] - 0.8 * X[:, 1]
y_logit = (rng.logistic(size=n) < xb).astype(float)
y_probit = (rng.normal(size=n) < xb).astype(float)


def proches(a, b, tol=1e-6):
    return np.allclose(a, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("lien", ["logit", "probit"])
def test_binaire_contre_statsmodels(lien):
    y = y_logit if lien == "logit" else y_probit
    mine = getattr(mle, lien)(y, X)
    ref = getattr(sm, lien.capitalize())(y, XC).fit(disp=0)
    assert proches(mine.theta, ref.params)
    assert proches(mine.se, ref.bse)
    assert proches(mine.loglik, ref.llf)
    assert proches(mine.pseudo_r2, ref.prsquared)


def test_logit_covariance_robuste():
    ref = sm.Logit(y_logit, XC).fit(disp=0, cov_type="HC0")
    assert proches(mle.logit(y_logit, X, cov="robuste").se, ref.bse)


@pytest.mark.parametrize("lien", ["logit", "probit"])
def test_effets_marginaux_binaire(lien):
    y = y_logit if lien == "logit" else y_probit
    mine = mle.effets_marginaux(getattr(mle, lien)(y, X))
    ref = getattr(sm, lien.capitalize())(y, XC).fit(disp=0).get_margeff(at="overall", dummy=True)
    assert proches(mine.valeurs, ref.margeff)
    assert proches(mine.se, ref.margeff_se)


# Newton et BFGS doivent arriver au meme optimum, meme depuis un point de depart eloigne
def test_newton_et_bfgs():
    f = lambda t: -np.sum(mle.modeles.ll_logit(t, y_logit, XC))
    g = lambda t: -mle.modeles.scores_logit(t, y_logit, XC).sum(axis=0)
    h = lambda t: -mle.modeles.hessien_logit(t, y_logit, XC)
    depart = np.array([3.0, -3.0, 3.0])
    tn, _, ok_n = newton(f, g, h, depart)
    tb, _, ok_b = bfgs(f, g, depart)
    assert ok_n and ok_b
    assert proches(tn, tb, 1e-4)


def test_hessien_numerique_egal_analytique():
    sans = mle.estimation.estimer(
        lambda t: mle.modeles.ll_logit(t, y_logit, XC), np.zeros(3), ["a", "b", "c"], methode="newton"
    )
    avec = mle.logit(y_logit, X)
    assert proches(sans.se, avec.se, 1e-5)


def test_ordonne_contre_statsmodels():
    lat = X @ [1.0, -0.8] + rng.logistic(size=n)
    y = np.digitize(lat, [-1, 0.5, 1.5])
    mine = mle.ordonne(y, X)
    ref = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=0, gtol=1e-9, maxiter=1000)
    seuils = np.concatenate([[ref.params[2]], ref.params[2] + np.cumsum(np.exp(ref.params[3:]))])
    assert proches(mine.theta[:2], ref.params[:2], 1e-5)
    assert proches(mine.theta[2:], seuils, 1e-5)
    assert proches(mine.se[:2], ref.bse[:2], 1e-4)
    assert proches(mine.loglik, ref.llf, 1e-8)
    p = sum(mine.moyenne(mine.theta, X, categorie=c) for c in range(4))
    assert proches(p, 1.0)


def test_multinomial_contre_statsmodels():
    U = np.column_stack([np.zeros(n), X @ [0.5, -1.0] + 0.3, X @ [-0.7, 0.4] - 0.2])
    P = np.exp(U) / np.exp(U).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in P])
    mine = mle.multinomial(y, X)
    ref = sm.MNLogit(y, XC).fit(disp=0)
    assert proches(mine.theta.reshape(2, 3), np.asarray(ref.params).T, 1e-5)
    assert proches(mine.se.reshape(2, 3), np.asarray(ref.bse).T, 1e-4)
    me = mle.effets_marginaux(mine, categorie=1)
    ref_me = ref.get_margeff(at="overall", dummy=True)
    assert proches(me.valeurs, ref_me.margeff[:, 1], 1e-5)
    assert proches(me.se, ref_me.margeff_se[:, 1], 1e-4)


# Tobit ecrit autrement dans statsmodels, pour verifier optimiseur et variance
class TobitRef(GenericLikelihoodModel):
    def loglikeobs(self, params):
        b, s = params[:-1], np.exp(params[-1])
        xb = self.exog @ b
        cens = self.endog <= 0
        return np.where(cens, norm.logcdf(-xb / s), norm.logpdf((self.endog - xb) / s) - np.log(s))


def donnees_tobit(graine=4, n=3000):
    r = np.random.default_rng(graine)
    x = r.normal(size=n)
    return x, np.maximum(1 + 2 * x + 1.5 * r.normal(size=n), 0)


def test_tobit_contre_statsmodels():
    x, y = donnees_tobit()
    mine = mle.tobit(y, x)
    ref = TobitRef(y, sm.add_constant(x)).fit(start_params=[1, 2, 0.4], disp=0, method="bfgs", gtol=1e-9)
    assert proches(mine.theta[:2], ref.params[:2], 1e-4)
    assert proches(mine.theta[2], np.exp(ref.params[2]), 1e-4)
    assert proches(mine.loglik, ref.llf, 1e-8)


# Sans observation censuree, le Tobit redonne les MCO, et sigma est l'ecart-type MV
def test_tobit_sans_censure_egal_mco():
    x, _ = donnees_tobit()
    y = 5 + 2 * x + np.random.default_rng(1).normal(size=len(x))
    mine = mle.tobit(y, x, gauche=-1e9)
    ref = sm.OLS(y, sm.add_constant(x)).fit()
    assert proches(mine.theta[:2], ref.params, 1e-5)
    assert proches(mine.theta[2], np.sqrt(ref.ssr / len(y)), 1e-5)


# Pour un Tobit censure en 0 : effet marginal sur E[y] = b * Phi(x'b / sigma)
def test_tobit_effet_marginal_formule_fermee():
    x, y = donnees_tobit()
    res = mle.tobit(y, x)
    attendu = res.theta[1] * np.mean(norm.cdf(res.X @ res.theta[:2] / res.theta[2]))
    assert proches(mle.effets_marginaux(res).valeurs[0], attendu, 1e-6)


# Heckman : les MCO sur l'echantillon selectionne sont biaises, pas le MV
def test_heckman_corrige_le_biais_de_selection():
    r = np.random.default_rng(2)
    m = 20000
    x, w, e = r.normal(size=m), r.normal(size=m), r.normal(size=m)
    u = 0.6 * e + 0.8 * r.normal(size=m)
    sel = (0.3 + 0.8 * x + 1.0 * w + u) > 0
    y = 1 + 2 * x + e
    res = mle.heckman(y, x, sel, np.column_stack([x, w]))
    mco = sm.OLS(y[sel], sm.add_constant(x[sel])).fit()
    vrai = np.array([1, 2, 0.3, 0.8, 1.0, 1.0, 0.6])
    assert res.converge
    assert np.allclose(res.theta, vrai, atol=0.06)
    assert abs(mco.params[0] - 1) > 0.1


def test_tests_wald_et_rapport_de_vraisemblance():
    complet = mle.logit(y_logit, X)
    restreint = mle.logit(y_logit, X[:, :1])
    ref_c = sm.Logit(y_logit, XC).fit(disp=0)
    ref_r = sm.Logit(y_logit, XC[:, :2]).fit(disp=0)
    stat, _ = mle.rapport_vraisemblance(restreint, complet)
    assert proches(stat, 2 * (ref_c.llf - ref_r.llf))
    stat_w, _ = mle.wald(complet, [0, 0, 1])
    assert proches(stat_w, (ref_c.params[2] / ref_c.bse[2]) ** 2)
