import time

import numpy as np
import statsmodels.api as sm
from scipy.special import expit

import bayes as by
from bayes import coeurs

rng = np.random.default_rng(0)
print("coeurs disponibles :", by.disponibles())

# 1. Panel a effets aleatoires : 200 individus observes 6 fois, vrais parametres
# b = (1, 2, -1), sigma2 = 1, tau2 = 0.64.
N, T = 200, 6
ids = np.repeat(np.arange(N), T)
X = rng.normal(size=(N * T, 2))
effets = 0.8 * rng.normal(size=N)
y = 1 + X @ [2, -1] + effets[ids] + rng.normal(size=N * T)

debut = time.time()
res = by.gibbs_panel(y, X, ids, n_iter=4000, burn=1000, noms=["x1", "x2"])
print(f"\nGIBBS, PANEL A EFFETS ALEATOIRES (coeur {res.backend}, {time.time() - debut:.2f} s)")
print(res.resume())
mm = sm.MixedLM(y, sm.add_constant(X), ids).fit(reml=True)
print("MixedLM (REML) : coefficients", np.round(mm.fe_params, 3), "sigma2 %.3f" % mm.scale,
      "tau2 %.3f" % np.asarray(mm.cov_re)[0, 0])
print("correlation entre effets estimes et vrais effets : %.3f" % np.corrcoef(res.alpha_moyen, effets)[0, 1])

# 2. Logit bayesien par Metropolis-Hastings, 3 regresseurs, vrais coefficients (0.5, 1, -0.5, 0.3)
n = 2000
Xl = rng.normal(size=(n, 3))
yl = (rng.random(n) < expit(0.5 + Xl @ [1, -0.5, 0.3])).astype(float)
debut = time.time()
mh = by.mh_logit(yl, Xl, n_iter=20000, burn=2000, prior_sd=10.0, noms=["x1", "x2", "x3"])
print(f"\nMETROPOLIS-HASTINGS, LOGIT (coeur {mh.backend}, {time.time() - debut:.2f} s)")
print(mh.resume())
print("taux d'acceptation par chaine :", np.round(mh.acceptation, 3))
ml = sm.Logit(yl, sm.add_constant(Xl)).fit(disp=0)
print("maximum de vraisemblance :", np.round(ml.params, 4), "\nerreurs standard         :", np.round(ml.bse, 4))

# 3. Les trois implementations produisent la meme chaine, a l'arrondi pres.
grp = np.repeat(np.arange(30), 4)
Xp = np.column_stack([np.ones(120), rng.normal(size=(120, 2))])
yp = Xp @ [1, 2, -1] + 0.8 * rng.normal(size=30)[grp] + 0.7 * rng.normal(size=120)
hyp = np.array([1e4, 0.01, 0.01, 1.0, 1.0])
init = np.concatenate([np.linalg.lstsq(Xp, yp, rcond=None)[0], np.zeros(30), [0.5, 1.0]])
chaines = {b: coeurs.gibbs(b, yp, Xp, grp, 30, init, hyp, 1000, 200, 1, 42)[0] for b in by.disponibles()}
print("\nECART MAXIMAL ENTRE LES CHAINES DE GIBBS (meme graine)")
for b in by.disponibles():
    print(f"  {b:>7} contre python : {np.abs(chaines[b] - chaines['python']).max():.1e}")
