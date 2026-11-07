import time

import numpy as np

import series as sr

# Filtre de Kalman sur un modele a 4 etats, 2 variables observees : C++ contre numpy.
rng = np.random.default_rng(0)
m, q = 4, 2
T = 0.6 * np.eye(m) + 0.1 * np.eye(m, k=1)
mats = dict(Z=rng.normal(size=(q, m)), H=0.5 * np.eye(q), T=T, Q=np.eye(m))

print(f"{'n':>8}{'numpy (s)':>12}{'C++ (s)':>10}{'gain':>8}")
for n in (1000, 10000, 100000):
    y = rng.normal(size=(n, q))
    debut = time.time()
    a = sr.filtre(y, **mats, stocker=False, moteur="numpy")
    t_numpy = time.time() - debut
    debut = time.time()
    b = sr.filtre(y, **mats, stocker=False, moteur="cpp")
    t_cpp = time.time() - debut
    assert np.isclose(a.loglik, b.loglik)
    print(f"{n:>8}{t_numpy:>12.3f}{t_cpp:>10.4f}{t_numpy / t_cpp:>8.0f}")
