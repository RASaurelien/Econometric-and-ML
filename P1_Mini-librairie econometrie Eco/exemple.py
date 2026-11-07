import numpy as np

from econometrie import mco, mc2e, gmm, hausman, j_hansen, breusch_pagan

# Un regresseur x est correle avec l'erreur u, donc le MCO est biaise.
# Les deux instruments z1 et z2 sont valides. Le vrai coefficient de x vaut 2.
VRAI = 2


def simuler(rng, n=500):
    z = rng.normal(size=(n, 2))
    u = rng.normal(size=n)
    x = z @ [1, 0.5] + 0.8 * u + rng.normal(size=n)
    y = 1 + VRAI * x + u
    return y, x, z


rng = np.random.default_rng(0)
y, x, z = simuler(rng)

res_mco = mco(y, x, noms=["const", "x"])
res_iv = mc2e(y, x, z, noms=["const", "x"])
res_gmm = gmm(y, x, z, noms=["const", "x"])

print("MCO")
print(res_mco)
print("\n2SLS")
print(res_iv)
print("\nGMM")
print(res_gmm)

print("\nHausman (x exogene ?) stat = %.2f, p = %.4f" % hausman(res_iv, res_mco, indices=[1]))
print("J de Hansen (instruments valides ?) stat = %.2f, p = %.3f" % j_hansen(res_gmm))
print("Breusch-Pagan sur le MCO : stat = %.2f, p = %.3f" % breusch_pagan(res_mco))

# Petit Monte Carlo : 300 echantillons, on regarde le biais moyen sur x
estimations = {"MCO": [], "2SLS": [], "GMM": []}
for _ in range(300):
    y, x, z = simuler(rng)
    estimations["MCO"].append(mco(y, x).beta[1])
    estimations["2SLS"].append(mc2e(y, x, z).beta[1])
    estimations["GMM"].append(gmm(y, x, z).beta[1])

print("\nMonte Carlo, vraie valeur = %d" % VRAI)
for nom, valeurs in estimations.items():
    print(f"{nom:>5} : moyenne = {np.mean(valeurs):.3f}, ecart-type = {np.std(valeurs):.3f}")
