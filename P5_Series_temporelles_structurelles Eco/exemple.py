import numpy as np

import series as sr
from series import structurel

rng = np.random.default_rng(0)

# 1. VAR(2) a trois variables : choix de l'ordre, causalite de Granger, IRF de Cholesky
T, k = 400, 3
A1 = np.array([[0.5, 0.1, 0.0], [0.2, 0.3, 0.1], [0.0, 0.1, 0.4]])
Y = np.zeros((T, k))
for t in range(2, T):
    Y[t] = 0.2 + A1 @ Y[t - 1] + 0.1 * Y[t - 2] + rng.normal(size=k)

ics, ordres = sr.choisir_ordre(Y, 6)
print("VAR : ordre retenu", ordres)
res = sr.var(Y, 2)
print("stable :", res.stable())
print("Granger, y1 -> y2 : F = %.2f, p = %.4f" % res.granger([0], [1]))
print("Granger, y3 -> y1 : F = %.2f, p = %.4f" % res.granger([2], [0]))

reponses = sr.irf_var(res, 8)
bas, haut = sr.irf_bootstrap(res, 8, B=500, graine=1)
print("\nReponse de y2 a un choc sur y1 (Cholesky), intervalle bootstrap a 95 %")
print("   h   IRF   bas  haut")
for h in range(0, 9, 2):
    print(f"{h:>4}{reponses[h, 1, 0]:>6.2f}{bas[h, 1, 0]:>6.2f}{haut[h, 1, 0]:>6.2f}")
part = sr.fevd_var(res, 10)[-1]
print("\nDecomposition de variance de y1 a 10 pas :", np.round(part[0], 2))

# 2. Blanchard-Quah : la vraie matrice d'impact est retrouvee avec une identification de long terme
n = 20000
A = np.array([[0.4, 0.2], [0.1, 0.3]])
L = np.array([[1.0, 0.0], [0.5, 0.8]])
B0 = (np.eye(2) - A) @ L
Z = np.zeros((n, 2))
e = rng.normal(size=(n, 2))
for t in range(1, n):
    Z[t] = A @ Z[t - 1] + B0 @ e[t]
est = sr.var(Z, 1, constante=False)
print("\nBlanchard-Quah, matrice d'impact vraie puis estimee")
print(np.round(B0, 3))
print(np.round(structurel.impact_long_terme(est.A, est.sigma), 3))

# 3. Cointegration : 3 variables, un vecteur (1, -1, 0.5) et des ajustements alpha = (-0.3, 0.1, 0)
beta, alpha = np.array([1.0, -1.0, 0.5]), np.array([-0.3, 0.1, 0.0])
W = np.zeros((500, 3))
for t in range(1, 500):
    W[t] = W[t - 1] + alpha * (W[t - 1] @ beta) + rng.normal(size=3)
print("\nTest de Johansen")
test = sr.johansen(W, p=2)
print(test)
print("rang retenu a 5 % :", test.rang())
v = sr.vecm(W, p=2, r=1)
print("beta estime :", np.round(v.beta.ravel(), 3), "(vrai : 1, -1, 0.5)")
print("alpha estime :", np.round(v.alpha.ravel(), 3), "(vrai : -0.3, 0.1, 0)")

# 4. Filtre de Kalman : niveau local avec 20 % de valeurs manquantes, puis ARMA(1,1)
niveau = np.cumsum(0.5 * rng.normal(size=300))
y = niveau + rng.normal(size=300)
y[rng.random(300) < 0.2] = np.nan
loc = sr.niveau_local(y)
print("\nNiveau local (vraies variances : 1 et 0.25)")
print(loc)
lisse, _ = loc.lisse()
print("erreur quadratique moyenne du niveau lisse : %.3f (observations brutes : %.3f)" % (
    np.mean((lisse[:, 0] - niveau) ** 2), np.nanmean((y - niveau) ** 2)))

e = rng.normal(size=800)
x = np.zeros(800)
for t in range(1, 800):
    x[t] = 0.6 * x[t - 1] + e[t] + 0.4 * e[t - 1]
print("\nARMA(1,1) (vrais parametres : ar = 0.6, ma = 0.4, sigma2 = 1)")
print(sr.arma(x, 1, 1))
print("\nmoteur C++ disponible :", sr.cpp_disponible())
