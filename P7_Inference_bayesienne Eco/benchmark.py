import time

import numpy as np
from scipy.special import expit

import bayes as by
from bayes import coeurs

rng = np.random.default_rng(0)
backends = by.disponibles()


# Chronometre une fonction trois fois et garde le meilleur temps.
def temps(f, repetitions=3):
    meilleur = np.inf
    for _ in range(repetitions):
        debut = time.time()
        sortie = f()
        meilleur = min(meilleur, time.time() - debut)
    return meilleur, sortie


# Gibbs : 500 individus observes 10 fois, 5 regresseurs, 2000 iterations.
N, T, k = 500, 10, 5
grp = np.repeat(np.arange(N), T)
X = np.column_stack([np.ones(N * T), rng.normal(size=(N * T, k - 1))])
y = X @ np.arange(1, k + 1) + 0.8 * rng.normal(size=N)[grp] + rng.normal(size=N * T)
hyp = np.array([1e4, 0.01, 0.01, 1.0, 1.0])
init = np.concatenate([np.linalg.lstsq(X, y, rcond=None)[0], np.zeros(N), [1.0, 1.0]])

print(f"GIBBS (n = {N * T}, k = {k}, {N} individus, 2000 iterations)")
print(f"{'coeur':>8}{'temps (s)':>12}{'gain':>8}")
resultats = {}
for b in backends[::-1]:
    t, sortie = temps(lambda: coeurs.gibbs(b, y, X, grp, N, init, hyp, 2000, 500, 1, 1), 1 if b == "python" else 3)
    resultats[b] = (t, sortie[0])
for b in backends:
    t = resultats[b][0]
    print(f"{b:>8}{t:>12.3f}{resultats['python'][0] / t:>8.0f}")
print("chaines identiques :", all(np.allclose(resultats[b][1], resultats["python"][1], atol=1e-8) for b in backends))

# Metropolis-Hastings : logit avec 5 regresseurs, 20000 iterations, pour plusieurs tailles d'echantillon.
# Chaque iteration evalue exp et log1p pour chaque observation.
L = 0.03 * np.eye(k)
print("\nMETROPOLIS-HASTINGS (k = 5, 20000 iterations), gain par rapport a python")
print(f"{'n':>8}" + "".join(f"{b + ' (s)':>12}" for b in backends) + "".join(f"{'gain ' + b:>12}" for b in backends[:-1]))
for n in (200, 1000, 5000):
    Xl = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    yl = (rng.random(n) < expit(Xl @ np.array([0.5, 1, -0.5, 0.3, 0.2]))).astype(float)
    t = {}
    for b in backends[::-1]:
        t[b], sortie = temps(lambda: coeurs.mh(b, yl, Xl, 10.0, L, np.zeros(k), 20000, 2000, 1, 1), 1 if b == "python" else 3)
    print(f"{n:>8}" + "".join(f"{t[b]:>12.3f}" for b in backends) + "".join(f"{t['python'] / t[b]:>12.1f}" for b in backends[:-1]))
