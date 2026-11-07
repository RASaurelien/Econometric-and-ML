import numpy as np

from bus_model import BusModel, EULER
from nfxp import fit_nfxp, estimate_transitions, state_counts
from ccp import fit_ccp, fit_npl, hotz_miller, pseudo_likelihood_step
from rl_agent import q_learning, replace_prob_from_q


def test_value_function_is_fixed_point():
    m = BusModel()
    V, v0, v1 = m.solve()
    mx = np.maximum(v0, v1)
    again = EULER + mx + np.log(np.exp(v0 - mx) + np.exp(v1 - mx))
    assert np.abs(again - V).max() < 1e-9


def test_hotz_miller_exact_with_true_ccp():
    m = BusModel()
    p1 = m.replace_prob()
    N = np.ones(m.n)
    theta = hotz_miller(p1, N, m.beta, m.trans, m.scale, m.n)
    assert abs(theta[0] - m.theta1) < 1e-6
    assert abs(theta[1] - m.rc) < 1e-6


def test_pseudo_likelihood_exact_with_true_ccp():
    m = BusModel()
    p1 = m.replace_prob()
    _, pi = m.stationary_replacement_rate()
    N = pi * 1e9
    theta, p_new = pseudo_likelihood_step(p1, N, N * p1, m.beta, m.trans, m.scale, m.n)
    assert abs(theta[0] - m.theta1) < 1e-3
    assert abs(theta[1] - m.rc) < 1e-3
    assert np.abs(p_new - p1).max() < 1e-4


def test_simulated_rate_matches_stationary_rate():
    m = BusModel()
    states, actions = m.simulate(300, 300, np.random.default_rng(0))
    rate, _ = m.stationary_replacement_rate()
    assert abs(actions.mean() - rate) < 0.003


def test_transition_estimates():
    m = BusModel()
    states, actions = m.simulate(300, 200, np.random.default_rng(1))
    assert np.abs(estimate_transitions(states, actions, m.n) - m.trans).max() < 0.01


def test_nfxp_and_ccp_recover_parameters():
    m = BusModel()
    states, actions = m.simulate(300, 200, np.random.default_rng(2))
    trans = estimate_transitions(states, actions, m.n)
    nf = fit_nfxp(states, actions, m.beta, trans, m.scale, m.n)
    cc = fit_ccp(states, actions, m.beta, trans, m.scale, m.n)
    npl = fit_npl(states, actions, m.beta, trans, m.scale, m.n)
    print("NFXP", nf["theta"], nf["se"], "HM", cc["theta"], "NPL", npl["theta"])
    assert np.all(np.abs(npl["theta"] - [m.theta1, m.rc]) < 3 * nf["se"] + 0.3)
    assert nf["success"]
    assert np.all(np.abs(nf["theta"] - [m.theta1, m.rc]) < 3 * nf["se"])
    assert np.all(np.abs(cc["theta"] - [m.theta1, m.rc]) < 3 * nf["se"] + 0.3)


def test_q_learning_close_to_dp_policy():
    m = BusModel()
    Q, visits = q_learning(m, n_steps=400000, seed=0)
    p_q, p_dp = replace_prob_from_q(Q), m.replace_prob()
    w = visits.sum(1) / visits.sum()
    err = np.sqrt(np.sum(w * (p_q - p_dp) ** 2))
    print("ecart pondere Q-learning vs DP", err)
    assert err < 0.05


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
