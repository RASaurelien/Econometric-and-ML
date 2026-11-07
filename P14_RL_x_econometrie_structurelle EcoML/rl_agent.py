import numpy as np


def q_learning(model, n_steps=600000, seed=0, c=50.0):
    """Q-learning tabulaire dans le simulateur du modele, avec chocs de Gumbel observes
    par l'agent (comme dans le modele logit). Q(x, a) approche la valeur de l'action a
    sans le choc courant, donc v0 et v1 du modele structurel."""
    rng = np.random.default_rng(seed)
    n, beta = model.n, model.beta
    u0 = model.flow_keep()
    Q = np.zeros((n, 2))
    visits = np.zeros((n, 2))
    shocks = rng.gumbel(size=(n_steps + 1, 2))
    inc = rng.choice(len(model.trans), size=n_steps, p=model.trans)
    explore = np.linspace(0.2, 0.02, n_steps)
    explore_draw = rng.random(n_steps)
    random_action = rng.integers(0, 2, size=n_steps)

    x = 0
    for t in range(n_steps):
        e = shocks[t]
        if explore_draw[t] < explore[t]:
            a = random_action[t]
        else:
            a = 0 if Q[x, 0] + e[0] >= Q[x, 1] + e[1] else 1
        reward = u0[x] if a == 0 else -model.rc
        nxt = min((0 if a == 1 else x) + inc[t], n - 1)
        e2 = shocks[t + 1]
        target = reward + beta * max(Q[nxt, 0] + e2[0], Q[nxt, 1] + e2[1])
        visits[x, a] += 1
        Q[x, a] += (c / (c + visits[x, a])) * (target - Q[x, a])
        x = nxt
    return Q, visits


def replace_prob_from_q(Q):
    return 1 / (1 + np.exp(Q[:, 0] - Q[:, 1]))
