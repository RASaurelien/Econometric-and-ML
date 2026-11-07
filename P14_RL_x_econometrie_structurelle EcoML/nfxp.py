import time
import numpy as np
from scipy.optimize import minimize

from bus_model import BusModel


def state_counts(states, actions, n_states):
    s, a = states.ravel(), actions.ravel()
    N = np.bincount(s, minlength=n_states).astype(float)
    R = np.bincount(s, weights=a, minlength=n_states)
    return N, R


def estimate_transitions(states, actions, n_states):
    """Frequences des increments de kilometrage (0, 1, 2), estimees en premiere etape."""
    prev, nxt = states[:, :-1], states[:, 1:]
    inc = nxt - np.where(actions[:, :-1] == 1, 0, prev)
    ok = (prev < n_states - 3) & (inc >= 0) & (inc <= 2)
    counts = np.bincount(inc[ok], minlength=3).astype(float)
    return counts / counts.sum()


def neg_loglik(theta, N, R, beta, trans, scale, n_states, cache):
    model = BusModel(theta[0], theta[1], beta, n_states, trans, scale)
    V, v0, v1 = model.solve(cache.get("V"))
    cache["V"] = V
    d = v0 - v1
    logp1 = -np.logaddexp(0, d)
    logp0 = -np.logaddexp(0, -d)
    return -(R @ logp1 + (N - R) @ logp0)


def numerical_hessian(f, x, step=1e-3):
    k = len(x)
    H = np.zeros((k, k))
    h = step * np.maximum(1.0, np.abs(x))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h[i], np.eye(k)[j] * h[j]
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def fit_nfxp(states, actions, beta, trans, scale, n_states, start=(1.0, 5.0)):
    """Maximum de vraisemblance avec boucle interne de point fixe (Rust, 1987)."""
    N, R = state_counts(states, actions, n_states)
    cache = {}
    f = lambda th: neg_loglik(th, N, R, beta, trans, scale, n_states, cache)
    t0 = time.time()
    res = minimize(f, np.array(start), method="L-BFGS-B", bounds=[(1e-3, None), (1e-3, None)],
                   options={"ftol": 1e-12, "gtol": 1e-8})
    H = numerical_hessian(f, res.x)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return {"theta": res.x, "se": se, "loglik": -res.fun, "time": time.time() - t0,
            "success": bool(res.success)}
