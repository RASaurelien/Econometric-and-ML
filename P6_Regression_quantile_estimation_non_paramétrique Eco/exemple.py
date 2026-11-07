import numpy as np
import statsmodels.api as sm
from statsmodels.datasets import engel

import quantnp as qn

rng = np.random.default_rng(0)

# 1. Courbe d'Engel (235 menages belges) : depense alimentaire en fonction du revenu.
# Les pentes varient fortement le long de la distribution, ce que les MCO ne voient pas.
donnees = engel.load_pandas().data
y, x = donnees.foodexp.values, donnees.income.values
print("REGRESSION QUANTILE, DEPENSE ALIMENTAIRE SUR LE REVENU")
mco = sm.OLS(y, sm.add_constant(x)).fit()
print(f"MCO : constante = {mco.params[0]:.2f}, pente = {mco.params[1]:.4f}")
print(f"{'tau':>6}{'constante':>11}{'pente':>9}{'se pente':>10}{'pseudo R2':>11}")
for r in qn.regressions_quantiles(y, x, (0.1, 0.25, 0.5, 0.75, 0.9)):
    print(f"{r.tau:>6}{r.beta[0]:>11.2f}{r.beta[1]:>9.4f}{r.se[1]:>10.4f}{r.pseudo_r2:>11.3f}")
stat, p = qn.egalite_pentes(y, x, (0.1, 0.9), B=300, graine=0)
print("Egalite des pentes aux quantiles 0.1 et 0.9 : chi2 = %.1f, p = %.1e" % (stat, p))

# 2. Estimation non parametrique : fenetre par validation croisee, effet de bord
n = 300
xs = np.sort(rng.uniform(0, 3, n))
ys = np.sin(2 * xs) + (0.2 + 0.2 * xs) * rng.normal(size=n)
h0, h1 = qn.choisir_h(xs, ys, degre=0), qn.choisir_h(xs, ys, degre=1)
print("\nFENETRES PAR VALIDATION CROISEE (Nadaraya-Watson, lineaire local) : %.3f, %.3f" % (h0, h1))
grille = np.array([0.0, 0.75, 1.5, 2.25, 3.0])
fit, bas, haut = qn.intervalle(xs, ys, grille, h1)
print(f"{'x':>6}{'vrai':>8}{'estime':>8}{'bas':>8}{'haut':>8}")
for g, v, b, hh in zip(grille, np.sin(2 * grille), fit, zip(bas, haut)):
    print(f"{g:>6.2f}{v:>8.3f}{b:>8.3f}{hh[0]:>8.3f}{hh[1]:>8.3f}")

# Biais de bord : sur une vraie droite, le lineaire local est exact, pas Nadaraya-Watson.
biais = {0: [], 1: []}
for _ in range(300):
    x = rng.uniform(0, 1, 200)
    y_ = 1 + 2 * x + 0.3 * rng.normal(size=200)
    for degre in (0, 1):
        biais[degre].append(qn.lisser(x, y_, [0.0], 0.15, "gaussien", degre)[0] - 1.0)
print("\nBiais en x = 0 d'une vraie droite (h = 0.15) : Nadaraya-Watson %.3f, lineaire local %.3f" % (
    np.mean(biais[0]), np.mean(biais[1])))

# 3. Modele partiellement lineaire : x depend de z, et z agit sur y de facon non lineaire.
m = 800
z = rng.uniform(0, 1, m)
xx = z + 0.5 * rng.normal(size=m)
yy = 1.5 * xx + np.sin(2 * np.pi * z) + 0.5 * rng.normal(size=m)
print("\nMODELE PARTIELLEMENT LINEAIRE (vrai coefficient = 1.5)")
naif = sm.OLS(yy, sm.add_constant(xx)).fit()
print(f"MCO sans g(z) : {naif.params[1]:.3f} (se {naif.bse[1]:.3f})")
rob = qn.robinson(yy, xx, z, noms=["x"])
print(f"Robinson      : {rob.beta[0]:.3f} (se {rob.se[0]:.3f})")
grille_z = np.array([0.25, 0.5, 0.75])
print("g(z) estime en 0.25, 0.5, 0.75 (centre) :", np.round(rob.g(grille_z) - rob.g(np.linspace(0, 1, 101)).mean(), 2),
      "vrai :", np.round(np.sin(2 * np.pi * grille_z), 2))
