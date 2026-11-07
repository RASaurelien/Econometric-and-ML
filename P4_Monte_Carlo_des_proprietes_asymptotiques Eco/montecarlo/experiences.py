import numpy as np
from scipy import stats

from . import bootstrap as bs
from . import simulation as sim
from .estimateurs import MCOBatch, IVBatch

VARIANTES = ["classique", "HC0", "HC1", "HC2", "HC3"]


# Met en forme un tableau texte. Les flottants sont affiches avec 3 decimales.
def tableau(entetes, lignes):
    cellules = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in ligne] for ligne in lignes]
    largeurs = [max(len(e), *(len(c[i]) for c in cellules)) for i, e in enumerate(entetes)]
    sortie = ["  ".join(e.rjust(w) for e, w in zip(entetes, largeurs))]
    sortie += ["  ".join(c.rjust(w) for c, w in zip(ligne, largeurs)) for ligne in cellules]
    return "\n".join(sortie)


# Taux de rejet de H0 : b1 = valeur, avec un test de Student (n - k ddl) pour
# les variantes de variance et un test bootstrap pour "wild" et "pairs".
def taux_rejet(X, y, b1, methodes, niveau, B, rng):
    obs = MCOBatch(y, X)
    crit = stats.t.ppf(1 - niveau / 2, obs.n - obs.k)
    sortie = {}
    for m in methodes:
        if m in VARIANTES:
            t = (obs.beta[:, 1] - b1) / obs.se(m)[:, 1]
            sortie[m] = np.mean(np.abs(t) > crit)
        else:
            sortie[m] = bs.rejets(X, y, 1, b1, m, B, niveau, "HC1", rng).mean()
    return sortie


# Taille du test (H0 vraie, niveau nominal 5 %) selon le scenario.
def taille(n=50, hetero=0.0, regresseurs="normal", erreurs="normale", R=2000, B=199,
           graine=0, methodes=VARIANTES + ["wild", "pairs"]):
    rng = np.random.default_rng(graine)
    X, y = sim.generer(R, n, rng, hetero=hetero, regresseurs=regresseurs, erreurs=erreurs)
    return taux_rejet(X, y, 0.5, methodes, 0.05, B, rng)


# Puissance : taux de rejet de H0 : b1 = 0 quand le vrai b1 vaut chaque effet.
def puissance(n=50, effets=(0.0, 0.1, 0.2, 0.3, 0.4), hetero=0.5, R=2000, B=199, graine=0,
              methodes=("classique", "HC3", "wild")):
    rng = np.random.default_rng(graine)
    lignes = []
    for effet in effets:
        X, y = sim.generer(R, n, rng, beta=(1.0, effet), hetero=hetero)
        r = taux_rejet(X, y, 0.0, methodes, 0.05, B, rng)
        lignes.append([effet] + [r[m] for m in methodes])
    return ["effet"] + list(methodes), lignes


# Biais et vitesse de convergence : MCO avec regresseur exogene, MCO avec regresseur
# endogene, 2SLS. sqrt(n) fois l'ecart-type doit se stabiliser quand n grandit.
def biais_consistance(ns=(25, 100, 400, 1600), R=5000, graine=0):
    rng = np.random.default_rng(graine)
    lignes = []
    for n in ns:
        y, x, _ = sim.generer_endogene(R, n, rng, endogene=False)
        exo = MCOBatch(y, x[..., None]).beta[:, 0] - 1
        y, x, z = sim.generer_endogene(R, n, rng, endogene=True)
        endo = MCOBatch(y, x[..., None]).beta[:, 0] - 1
        iv = IVBatch(y, x, z).deux_smc()[0] - 1
        lignes.append([n, exo.mean(), np.sqrt(n) * exo.std(), endo.mean(), iv.mean(), np.sqrt(n) * iv.std()])
    entetes = ["n", "biais MCO exo", "sqrt(n)*sd", "biais MCO endo", "biais 2SLS", "sqrt(n)*sd"]
    return entetes, lignes


# Normalite asymptotique de la statistique t robuste (HC3) avec erreurs asymetriques
# et heteroscedastiques : distance de Kolmogorov-Smirnov a la loi normale et taux de
# rejet quand on utilise la valeur critique normale de 5 %.
def normalite(ns=(10, 25, 100, 400), R=5000, graine=0):
    rng = np.random.default_rng(graine)
    lignes = []
    for n in ns:
        X, y = sim.generer(R, n, rng, hetero=0.3, erreurs="asymetrique")
        obs = MCOBatch(y, X)
        t = (obs.beta[:, 1] - 0.5) / obs.se("HC3")[:, 1]
        lignes.append([n, stats.kstest(t, "norm").statistic, stats.skew(t), np.mean(np.abs(t) > 1.96)])
    return ["n", "KS", "asymetrie de t", "rejet a 5 %"], lignes


# Instruments faibles : m instruments, force mesuree par le parametre de concentration mu2.
# On compare le biais median et la taille (niveau nominal 5 %) des 2SLS, du LIML et
# du test d'Anderson-Rubin.
def instruments_faibles(mu2s=(1, 5, 10, 30, 100), n=200, m=5, R=5000, graine=0):
    rng = np.random.default_rng(graine)
    f_ar = stats.f.ppf(0.95, m, n - m)
    lignes = []
    for mu2 in mu2s:
        y, x, z = sim.generer_iv(R, n, m, mu2, rng)
        iv = IVBatch(y, x, z)
        b2, se2 = iv.deux_smc()
        bl, sel = iv.liml()
        mco = MCOBatch(y, x[..., None]).beta[:, 0]
        lignes.append([
            mu2, iv.f_premiere_etape().mean(), np.median(mco - 1), np.median(b2 - 1), np.median(bl - 1),
            np.mean(np.abs((b2 - 1) / se2) > 1.96), np.mean(np.abs((bl - 1) / sel) > 1.96),
            np.mean(iv.anderson_rubin(1.0) > f_ar),
        ])
    entetes = ["mu2", "F moyen", "biais med. MCO", "biais med. 2SLS", "biais med. LIML",
               "rejet 2SLS", "rejet LIML", "rejet AR"]
    return entetes, lignes
