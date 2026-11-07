import time

import montecarlo as mc

debut = time.time()

# 1. Biais et consistance : le MCO endogene reste biaise (0.6 / 1.48 = 0.405),
# les 2SLS convergent, et sqrt(n) fois l'ecart-type se stabilise.
print("BIAIS ET CONSISTANCE (vrai beta = 1, 5000 replications)")
print(mc.tableau(*mc.biais_consistance()))

# 2. Normalite asymptotique avec des erreurs tres asymetriques.
print("\nNORMALITE DE LA STATISTIQUE T ROBUSTE (HC3)")
print(mc.tableau(*mc.normalite()))

# 3. Taille du test de H0 : b1 = 0.5, niveau nominal 5 %.
scenarios = [
    ("homoscedastique", dict(n=50)),
    ("heteroscedastique", dict(n=50, hetero=0.5)),
    ("heterosc. + leviers", dict(n=50, hetero=0.5, regresseurs="asymetrique")),
    ("heterosc. + leviers, n = 200", dict(n=200, hetero=0.5, regresseurs="asymetrique")),
]
methodes = mc.experiences.VARIANTES + ["wild", "pairs"]
lignes = []
for nom, kw in scenarios:
    r = mc.taille(**kw)
    lignes.append([nom] + [float(r[m]) for m in methodes])
print("\nTAILLE DU TEST (2000 replications, 199 tirages bootstrap)")
print(mc.tableau(["scenario"] + methodes, lignes))

# 4. Puissance, avec des erreurs heteroscedastiques.
print("\nPUISSANCE (n = 50, heteroscedasticite)")
print(mc.tableau(*mc.puissance()))

# 5. Instruments faibles : 5 instruments, n = 200.
print("\nINSTRUMENTS FAIBLES (vrai beta = 1, rho = 0.8, niveau nominal 5 %)")
print(mc.tableau(*mc.instruments_faibles()))

print("\nTemps total : %.0f s" % (time.time() - debut))
