import time

import numpy as np
import statsmodels.api as sm

import quantnp as qn

rng = np.random.default_rng(0)

# Regression quantile : programme lineaire (dual) contre IRLS de statsmodels, k = 3 regresseurs.
print("REGRESSION QUANTILE (mediane)")
print(f"{'n':>8}{'LP dual (s)':>13}{'IRLS statsmodels (s)':>22}")
for n in (1000, 10000, 100000):
    X = np.column_stack([rng.normal(size=(n, 2))])
    y = 1 + X @ [2, -1] + rng.standard_t(3, size=n)
    debut = time.time()
    qn.regression_quantile(y, X, 0.5, vcov="iid")
    t_lp = time.time() - debut
    debut = time.time()
    sm.QuantReg(y, sm.add_constant(X)).fit(q=0.5)
    t_irls = time.time() - debut
    print(f"{n:>8}{t_lp:>13.3f}{t_irls:>22.3f}")

# Validation croisee d'un lineaire local : cout en n^2 pour une fenetre donnee.
print("\nVALIDATION CROISEE, UNE FENETRE (lineaire local)")
print(f"{'n':>8}{'temps (s)':>12}")
for n in (500, 1000, 2000, 4000):
    x = rng.uniform(0, 1, n)
    y = np.sin(6 * x) + 0.3 * rng.normal(size=n)
    debut = time.time()
    qn.cv_loo(x, y, 0.05, "gaussien", 1)
    print(f"{n:>8}{time.time() - debut:>12.3f}")
