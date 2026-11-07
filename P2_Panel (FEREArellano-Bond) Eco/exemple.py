import numpy as np

import panel as pn
from panel.dynamique import arellano_bond

rng = np.random.default_rng(0)

# Panel statique : les effets individuels sont correles au regresseur x.
# Le vrai coefficient de x vaut 1. Les effets fixes doivent le retrouver,
# pas les MCO empiles ni les effets aleatoires.
N, T = 200, 6
ids = np.repeat(np.arange(N), T)
alpha = rng.normal(size=N)[ids]
x = rng.normal(size=N * T) + alpha
y = 2 + x + alpha + rng.normal(size=N * T)

empile = pn.mco_empile(y, x, ids, cov="cluster", noms=["x"])
fe = pn.effets_fixes(y, x, ids, cov="cluster", noms=["x"])
re = pn.effets_aleatoires(y, x, ids, noms=["x"])
mund = pn.mundlak(y, x, ids, noms=["x"])

print("coefficient de x (vrai = 1)")
print("  MCO empiles      : %.3f" % empile.beta[1])
print("  effets aleatoires : %.3f" % re.beta[1])
print("  effets fixes     : %.3f" % fe.beta[0])
print("  Mundlak          : %.3f" % mund.beta[1])
print("\nF effets individuels : stat = %.1f, p = %.4f" % pn.f_effets_fixes(empile, fe))
print("Hausman FE vs RE     : stat = %.1f, p = %.4f" % pn.hausman(fe, re))
print("Mundlak (robuste)    : stat = %.1f, p = %.4f" % pn.test_mundlak(mund))

# Panel dynamique : y_it = 0.5 y_i,t-1 + 0.8 x_it + alpha_i + e_it
# Avec T court, les effets fixes sous-estiment rho (biais de Nickell).
def simuler(N, T):
    a = rng.normal(size=N)
    x = rng.normal(size=(N, T))
    Y = np.zeros((N, T))
    Y[:, 0] = a + rng.normal(size=N)
    for t in range(1, T):
        Y[:, t] = 0.5 * Y[:, t - 1] + 0.8 * x[:, t] + a + rng.normal(size=N)
    return Y, x


N, T = 300, 6
Y, X = simuler(N, T)
print("\nArellano-Bond en deux etapes (vrai rho = 0.5, vrai beta = 0.8)")
print(arellano_bond(Y.ravel(), X.ravel(), np.repeat(np.arange(N), T), noms=["x"]))

# Monte Carlo : 200 echantillons, estimation de rho
rho_fe, rho_ab = [], []
for _ in range(200):
    Y, X = simuler(N, T)
    Xfe = np.column_stack([Y[:, :-1].ravel(), X[:, 1:].ravel()])
    rho_fe.append(pn.effets_fixes(Y[:, 1:].ravel(), Xfe, np.repeat(np.arange(N), T - 1)).beta[0])
    rho_ab.append(arellano_bond(Y.ravel(), X.ravel(), np.repeat(np.arange(N), T)).beta[0])

print("\nMonte Carlo, vrai rho = 0.5")
print("  effets fixes  : moyenne = %.3f, ecart-type = %.3f" % (np.mean(rho_fe), np.std(rho_fe)))
print("  Arellano-Bond : moyenne = %.3f, ecart-type = %.3f" % (np.mean(rho_ab), np.std(rho_ab)))
