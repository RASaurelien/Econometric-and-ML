import numpy as np
import statsmodels.api as sm

import mle

rng = np.random.default_rng(0)
n = 3000
x = rng.normal(size=n)
d = rng.integers(0, 2, n)
X = np.column_stack([x, d])

# Logit et probit, avec effets marginaux moyens
y = (rng.logistic(size=n) < 0.5 + x - 0.8 * d).astype(float)
res = mle.logit(y, X, noms=["x", "d"])
print("LOGIT")
print(res)
print("\nEffets marginaux moyens")
print(mle.effets_marginaux(res))

# Logit ordonne : 4 categories
lat = x - 0.8 * d + rng.logistic(size=n)
print("\nLOGIT ORDONNE")
print(mle.ordonne(np.digitize(lat, [-1, 0.5, 1.5]), X, noms=["x", "d"]))

# Tobit : 35 % des observations sont censurees a 0, vrais coefficients (1, 2)
ys = np.maximum(1 + 2 * x + 1.5 * rng.normal(size=n), 0)
print("\nTOBIT (vrais coefficients : 1, 2, sigma = 1.5)")
tob = mle.tobit(ys, x, noms=["x"])
print(tob)
mco = sm.OLS(ys, sm.add_constant(x)).fit()
print("MCO ignorant la censure : const = %.3f, x = %.3f" % tuple(mco.params))
print("Effet marginal moyen de x sur E[y] (MCO : %.3f)" % mco.params[1])
print(mle.effets_marginaux(tob))

# Heckman : y n'est observe que pour les selectionnes, et la selection depend
# d'un terme d'erreur correle a celui de y (rho = 0.6)
m = 4000
x, w, e = rng.normal(size=m), rng.normal(size=m), rng.normal(size=m)
u = 0.6 * e + 0.8 * rng.normal(size=m)
sel = (0.3 + 0.8 * x + 1.0 * w + u) > 0
yy = 1 + 2 * x + e
print("\nHECKMAN (vrais coefficients : y = 1, 2 ; rho = 0.6)")
print(mle.heckman(yy, x, sel, np.column_stack([x, w]), noms_x=["x"], noms_z=["x", "w"]))
biais = sm.OLS(yy[sel], sm.add_constant(x[sel])).fit()
print("MCO sur les selectionnes : const = %.3f, x = %.3f" % tuple(biais.params))
