import time

import numpy as np

import montecarlo as mc
from montecarlo import simulation as sim

R, n = 5000, 100
X, y = sim.generer(R, n, np.random.default_rng(0), hetero=0.5)


# Version naive : une regression et un sandwich HC3 par replication.
def boucle():
    sortie = np.empty(R)
    for r in range(R):
        XtX_inv = np.linalg.inv(X[r].T @ X[r])
        beta = XtX_inv @ X[r].T @ y[r]
        e = y[r] - X[r] @ beta
        h = np.einsum("ni,ij,nj->n", X[r], XtX_inv, X[r])
        meat = (X[r] * (e / (1 - h))[:, None] ** 2).T @ X[r]
        sortie[r] = np.sqrt((XtX_inv @ meat @ XtX_inv)[1, 1])
    return sortie


debut = time.time()
a = boucle()
t_boucle = time.time() - debut

debut = time.time()
b = mc.MCOBatch(y, X).se("HC3")[:, 1]
t_batch = time.time() - debut

print(f"{R} regressions avec sandwich HC3 (n = {n})")
print(f"boucle  : {t_boucle:.3f} s")
print(f"vectorise : {t_batch:.3f} s, acceleration x{t_boucle / t_batch:.0f}")
print("resultats identiques :", np.allclose(a, b))

# Bootstrap wild : 200 replications, 199 tirages chacune.
R2, B = 200, 199
X2, y2 = X[:R2], y[:R2]
w = np.random.default_rng(1).choice([-1.0, 1.0], size=(R2, B, n))


def boucle_bootstrap():
    sortie = np.empty((R2, B))
    for r in range(R2):
        xj = X2[r][:, 1]
        yr = y2[r] - 0.5 * xj
        e_r = yr - X2[r][:, 0] * yr.mean()
        ajuste = yr - e_r + 0.5 * xj
        XtX_inv = np.linalg.inv(X2[r].T @ X2[r])
        for b in range(B):
            ys = ajuste + e_r * w[r, b]
            beta = XtX_inv @ X2[r].T @ ys
            e = ys - X2[r] @ beta
            meat = (X2[r] * e[:, None] ** 2).T @ X2[r]
            sortie[r, b] = (beta[1] - 0.5) / np.sqrt(n / (n - 2) * (XtX_inv @ meat @ XtX_inv)[1, 1])
    return sortie


debut = time.time()
a = boucle_bootstrap()
t_boucle = time.time() - debut

debut = time.time()
b = mc.bootstrap.wild_t(X2, y2, 1, 0.5, w, "HC1")[1]
t_batch = time.time() - debut

print(f"\nwild bootstrap, {R2} replications x {B} tirages ({R2 * B} regressions)")
print(f"boucle  : {t_boucle:.3f} s")
print(f"vectorise : {t_batch:.3f} s, acceleration x{t_boucle / t_batch:.0f}")
print("resultats identiques :", np.allclose(a, b))
