import numpy as np

EULER = 0.5772156649


class BusModel:
    """Modele de remplacement de moteur de Rust (1987).
    Etat : kilometrage discretise (0 a n_states - 1). Action : garder (0) ou remplacer (1).
    Utilite de garder : -scale * theta1 * x. Utilite de remplacer : -rc.
    Apres un remplacement le kilometrage repart de zero avant l'increment de la periode."""

    def __init__(self, theta1=2.5, rc=10.0, beta=0.95, n_states=90,
                 trans=(0.35, 0.5, 0.15), scale=0.01):
        self.theta1 = theta1
        self.rc = rc
        self.beta = beta
        self.n = n_states
        self.trans = np.asarray(trans, dtype=float)
        self.scale = scale

    def keep_matrix(self):
        P = np.zeros((self.n, self.n))
        for x in range(self.n):
            for j, pj in enumerate(self.trans):
                P[x, min(x + j, self.n - 1)] += pj
        return P

    def flow_keep(self):
        return -self.scale * self.theta1 * np.arange(self.n)

    def solve(self, V0=None, tol=1e-12, max_iter=5000):
        """Iteration sur la fonction de valeur ex ante. Renvoie V, v0, v1."""
        P = self.keep_matrix()
        u0 = self.flow_keep()
        V = np.zeros(self.n) if V0 is None else V0.copy()
        for _ in range(max_iter):
            v0 = u0 + self.beta * P @ V
            v1 = -self.rc + self.beta * P[0] @ V
            m = np.maximum(v0, v1)
            V_new = EULER + m + np.log(np.exp(v0 - m) + np.exp(v1 - m))
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        v0 = u0 + self.beta * P @ V
        v1 = np.full(self.n, -self.rc + self.beta * P[0] @ V)
        return V, v0, v1

    def replace_prob(self, V0=None):
        V, v0, v1 = self.solve(V0)
        return 1 / (1 + np.exp(v0 - v1))

    def stationary_replacement_rate(self):
        p1 = self.replace_prob()
        P = self.keep_matrix()
        M = (1 - p1)[:, None] * P + p1[:, None] * P[0][None, :]
        w, vec = np.linalg.eig(M.T)
        pi = np.real(vec[:, np.argmax(np.real(w))])
        pi = pi / pi.sum()
        return float(pi @ p1), pi

    def simulate(self, n_buses, T, rng, burn_in=50):
        p1 = self.replace_prob()
        x = np.zeros(n_buses, dtype=int)
        states, actions = [], []
        for t in range(T + burn_in):
            a = (rng.random(n_buses) < p1[x]).astype(int)
            inc = rng.choice(len(self.trans), size=n_buses, p=self.trans)
            nxt = np.minimum(np.where(a == 1, 0, x) + inc, self.n - 1)
            if t >= burn_in:
                states.append(x.copy())
                actions.append(a)
            x = nxt
        return np.array(states).T, np.array(actions).T
