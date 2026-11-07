import numpy as np
import pandas as pd
import pytest
from linearmodels.iv import IVGMM
from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS

import panel as pn
from panel.dynamique import arellano_bond, construire, instruments_gmm, matrice_h

# Panel statique equilibre, avec des effets individuels correles aux x
rng = np.random.default_rng(1)
N, T = 120, 6
ids = np.repeat(np.arange(N), T)
temps = np.tile(np.arange(T), N)
alpha = rng.normal(size=N)[ids]
X = rng.normal(size=(N * T, 2)) + 0.5 * alpha[:, None]
y = 1 + X @ [1.0, -0.5] + alpha + rng.normal(size=N * T) * (1 + np.abs(X[:, 0]))

df = pd.DataFrame(
    np.column_stack([y, X]), columns=["y", "a", "b"],
    index=pd.MultiIndex.from_arrays([ids, temps]),
)
exog = df[["a", "b"]]
exog_c = exog.assign(const=1.0)[["const", "a", "b"]]


def test_effets_fixes_classique():
    ref = PanelOLS(df.y, exog, entity_effects=True).fit()
    mine = pn.effets_fixes(y, X, ids)
    assert np.allclose(mine.beta, ref.params.values)
    assert np.allclose(mine.se, ref.std_errors.values)


def test_effets_fixes_cluster():
    ref = PanelOLS(df.y, exog, entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True, group_debias=True
    )
    assert np.allclose(pn.effets_fixes(y, X, ids, cov="cluster").se, ref.std_errors.values)


def test_effets_fixes_desequilibre():
    garde = rng.random(N * T) > 0.2
    d = df[garde]
    ref = PanelOLS(d.y, d[["a", "b"]], entity_effects=True).fit()
    mine = pn.effets_fixes(y[garde], X[garde], ids[garde])
    assert np.allclose(mine.beta, ref.params.values)
    assert np.allclose(mine.se, ref.std_errors.values)


def test_mco_empile_cluster():
    ref = PanelOLS(df.y, exog_c).fit(
        cov_type="clustered", cluster_entity=True, group_debias=True
    )
    mine = pn.mco_empile(y, X, ids, cov="cluster")
    assert np.allclose(mine.beta, ref.params.values)
    assert np.allclose(mine.se, ref.std_errors.values)


def test_between():
    ref = BetweenOLS(df.y, exog_c).fit()
    mine = pn.between(y, X, ids)
    assert np.allclose(mine.beta, ref.params.values)
    assert np.allclose(mine.se, ref.std_errors.values)


def test_effets_aleatoires():
    ref = RandomEffects(df.y, exog_c).fit()
    mine = pn.effets_aleatoires(y, X, ids)
    assert np.allclose(mine.beta, ref.params.values)
    assert np.allclose(mine.se, ref.std_errors.values)


def test_f_effets_fixes():
    ref = PanelOLS(df.y, exog, entity_effects=True).fit()
    stat, p = pn.f_effets_fixes(pn.mco_empile(y, X, ids), pn.effets_fixes(y, X, ids))
    assert np.isclose(stat, ref.f_pooled.stat)
    assert np.isclose(p, ref.f_pooled.pval)


# Dans la regression de Mundlak, les pentes sont exactement celles des effets fixes
def test_mundlak_egal_effets_fixes():
    mund = pn.mundlak(y, X, ids)
    fe = pn.effets_fixes(y, X, ids)
    assert np.allclose(mund.beta[1:3], fe.beta)


# Les x sont correles aux effets individuels, les deux tests doivent rejeter
def test_hausman_et_mundlak_rejettent():
    fe, re = pn.effets_fixes(y, X, ids), pn.effets_aleatoires(y, X, ids)
    assert pn.hausman(fe, re)[1] < 0.05
    assert pn.test_mundlak(pn.mundlak(y, X, ids))[1] < 0.05


# Panel dynamique : y_it = 0.5 y_i,t-1 + 0.8 x_it + alpha_i + e_it
def simuler_dynamique(graine, N=300, T=7):
    r = np.random.default_rng(graine)
    a = r.normal(size=N)
    x = r.normal(size=(N, T))
    Y = np.zeros((N, T))
    Y[:, 0] = a + r.normal(size=N)
    for t in range(1, T):
        Y[:, t] = 0.5 * Y[:, t - 1] + 0.8 * x[:, t] + a + r.normal(size=N)
    return Y, x


def test_arellano_bond_contre_linearmodels():
    Y, x = simuler_dynamique(3)
    n_ind = Y.shape[0]
    id_ab = np.repeat(np.arange(n_ind), Y.shape[1])
    dep, R, dXe = construire(Y, x[:, :, None], 1)
    Z = np.concatenate([instruments_gmm(Y, 1, False), dXe], axis=2)
    n_eq = dep.shape[1]
    nobs = n_ind * n_eq
    W1 = np.linalg.inv(np.einsum("ira,rs,isb->ab", Z, matrice_h(n_eq), Z))
    groupes = np.repeat(np.arange(n_ind), n_eq)
    ref = IVGMM(
        dep.reshape(-1), None, R.reshape(nobs, -1), Z.reshape(nobs, -1),
        weight_type="clustered", clusters=groupes,
    )
    un = ref.fit(iter_limit=1, initial_weight=W1 * nobs, cov_type="clustered", clusters=groupes)
    deux = ref.fit(iter_limit=2, initial_weight=W1 * nobs, cov_type="clustered", clusters=groupes)

    m1 = arellano_bond(Y.ravel(), x.ravel(), id_ab, deux_etapes=False)
    m2 = arellano_bond(Y.ravel(), x.ravel(), id_ab)
    assert np.allclose(m1.beta, un.params.values)
    assert np.allclose(m1.se, un.std_errors.values)
    assert np.allclose(m2.beta, deux.params.values)
    assert np.isclose(m2.hansen[0], deux.j_stat.stat)


# Avec T = 3 et sans x, le modele est juste identifie : c'est l'estimateur d'Anderson-Hsiao
def test_arellano_bond_anderson_hsiao():
    Y, _ = simuler_dynamique(5, N=500, T=3)
    id_ab = np.repeat(np.arange(500), 3)
    dy1, dy2 = Y[:, 1] - Y[:, 0], Y[:, 2] - Y[:, 1]
    attendu = (Y[:, 0] * dy2).sum() / (Y[:, 0] * dy1).sum()
    res = arellano_bond(Y.ravel(), None, id_ab, deux_etapes=False)
    assert np.isclose(res.beta[0], attendu)


# Biais de Nickell : avec T court, les effets fixes sous-estiment rho, pas Arellano-Bond
def test_nickell():
    Y, x = simuler_dynamique(8, N=400, T=6)
    n_ind = Y.shape[0]
    id_ab = np.repeat(np.arange(n_ind), 6)
    ab = arellano_bond(Y.ravel(), x.ravel(), id_ab)
    ids_fe = np.repeat(np.arange(n_ind), 5)
    Xfe = np.column_stack([Y[:, :-1].ravel(), x[:, 1:].ravel()])
    fe = pn.effets_fixes(Y[:, 1:].ravel(), Xfe, ids_fe)
    assert abs(ab.beta[0] - 0.5) < abs(fe.beta[0] - 0.5)
