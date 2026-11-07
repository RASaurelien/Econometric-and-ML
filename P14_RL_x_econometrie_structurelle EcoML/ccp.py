import time
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from bus_model import BusModel, EULER
from nfxp import state_counts, estimate_transitions


def estimate_ccp(N, R, n_states, degree=3):
    """Probabilite de remplacement par etat : logit sur un polynome du kilometrage."""
    s = np.arange(n_states) / (n_states - 1)
    feats = np.column_stack([s ** k for k in range(1, degree + 1)])
    feats = (feats - feats.mean(0)) / feats.std(0)
    seen = N > 0
    X = np.vstack([feats[seen], feats[seen]])
    y = np.r_[np.ones(seen.sum()), np.zeros(seen.sum())]
    w = np.r_[R[seen], (N - R)[seen]]
    keep = w > 0
    clf = LogisticRegression(penalty=None, max_iter=5000).fit(X[keep], y[keep], sample_weight=w[keep])
    return np.clip(clf.predict_proba(feats)[:, 1], 1e-6, 1 - 1e-6)


def hotz_miller(p1, N, beta, trans, scale, n_states):
    """Inversion de Hotz et Miller. Avec renouvellement, v0 - v1 = ln(p0 / p1) s'ecrit
    -theta1 * scale * x + RC + beta * (w(x) - w(0)), avec w(x) = E[-ln p1(x') | x, garder].
    C'est donc une regression lineaire en (theta1, RC) une fois les CCP estimees."""
    P = BusModel(beta=beta, n_states=n_states, trans=trans, scale=scale).keep_matrix()
    w = P @ (-np.log(p1))
    y = np.log(1 - p1) - np.log(p1) - beta * (w - w[0])
    x = np.arange(n_states)
    A = np.column_stack([-scale * x, np.ones(n_states)])
    weight = np.sqrt(N * p1 * (1 - p1))
    use = weight > 0
    coef = np.linalg.lstsq(A[use] * weight[use, None], y[use] * weight[use], rcond=None)[0]
    return coef


def fit_ccp(states, actions, beta, trans, scale, n_states):
    t0 = time.time()
    N, R = state_counts(states, actions, n_states)
    p1 = estimate_ccp(N, R, n_states)
    theta = hotz_miller(p1, N, beta, trans, scale, n_states)
    return {"theta": theta, "p1": p1, "time": time.time() - t0}


def _value_components(p1, beta, trans, scale, n_states):
    """Valeur ex ante sous la politique p, ecrite comme V_c + theta1 * V_1 + RC * V_2
    (elle est lineaire en theta). Un seul systeme lineaire par composante, pas de point fixe."""
    P = BusModel(beta=beta, n_states=n_states, trans=trans, scale=scale).keep_matrix()
    p0 = 1 - p1
    M = p0[:, None] * P + p1[:, None] * P[0][None, :]
    inv = np.linalg.inv(np.eye(n_states) - beta * M)
    x = np.arange(n_states)
    s_c = p0 * (EULER - np.log(p0)) + p1 * (EULER - np.log(p1))
    s_1 = p0 * (-scale * x)
    s_2 = -p1
    D = P[0][None, :] - P
    return [beta * D @ (inv @ s) for s in (s_c, s_1, s_2)], scale * x


def pseudo_likelihood_step(p1, N, R, beta, trans, scale, n_states, start=(1.0, 5.0)):
    """Estimateur de pseudo-vraisemblance (Aguirregabiria et Mira, 2002) : les CCP ne servent
    qu'a calculer la valeur de continuation, la vraisemblance reste celle du logit."""
    (d_c, d_1, d_2), sx = _value_components(p1, beta, trans, scale, n_states)
    z1, z2 = sx + d_1, -1.0 + d_2

    def nll(theta):
        delta = d_c + theta[0] * z1 + theta[1] * z2
        logp1 = -np.logaddexp(0, -delta)
        logp0 = -np.logaddexp(0, delta)
        return -(R @ logp1 + (N - R) @ logp0)

    res = minimize(nll, np.array(start), method="L-BFGS-B", bounds=[(1e-3, None), (1e-3, None)])
    delta = d_c + res.x[0] * z1 + res.x[1] * z2
    return res.x, 1 / (1 + np.exp(-delta))


def fit_npl(states, actions, beta, trans, scale, n_states, n_iter=3):
    """NPL : on part des CCP du premier stade puis on alterne pseudo-vraisemblance et mise a jour des CCP."""
    t0 = time.time()
    N, R = state_counts(states, actions, n_states)
    p1 = estimate_ccp(N, R, n_states)
    for _ in range(n_iter):
        theta, p1 = pseudo_likelihood_step(np.clip(p1, 1e-6, 1 - 1e-6), N, R, beta, trans, scale, n_states)
    return {"theta": theta, "p1": p1, "time": time.time() - t0}


def bootstrap_ccp(states, actions, beta, scale, n_states, n_boot, rng, fit=None):
    n_buses = states.shape[0]
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_buses, n_buses)
        s, a = states[idx], actions[idx]
        trans = estimate_transitions(s, a, n_states)
        draws.append((fit or fit_ccp)(s, a, beta, trans, scale, n_states)["theta"])
    return np.std(draws, axis=0, ddof=1)
