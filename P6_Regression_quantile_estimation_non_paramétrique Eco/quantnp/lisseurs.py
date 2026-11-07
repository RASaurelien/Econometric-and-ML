import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from .noyaux import NOYAUX


# Fenetres de reference (Silverman et Scott) pour la densite.
def sigma_robuste(x):
    return min(np.std(x, ddof=1), np.subtract(*np.percentile(x, [75, 25])) / 1.349)


def bw_silverman(x):
    return 0.9 * sigma_robuste(x) * len(x) ** (-0.2)


def bw_scott(x):
    return 1.059 * sigma_robuste(x) * len(x) ** (-0.2)


# Estimateur a noyau de la densite : (1 / n h) somme K((g - x_i) / h).
def densite(x, grille, h=None, noyau="gaussien"):
    x, grille = np.asarray(x, dtype=float), np.asarray(grille, dtype=float)
    h = bw_silverman(x) if h is None else h
    return NOYAUX[noyau]((grille[:, None] - x[None, :]) / h).sum(axis=1) / (len(x) * h)


# Poids equivalents L (G, n) d'un polynome local : l'estimation en g vaut L[g] @ y.
# degre 0 : Nadaraya-Watson, degre 1 : lineaire local. On resout, pour chaque g, les MCO
# ponderes de y sur 1, (x_i - g), ..., (x_i - g)^degre, et on garde le coefficient de la constante.
# Si aucun point n'est dans le support du noyau, la ligne vaut nan.
def poids_locaux(x, grille, h, noyau="gaussien", degre=1):
    U = x[None, :] - np.asarray(grille, dtype=float)[:, None]
    W = NOYAUX[noyau](U / h)
    if degre == 0:
        S = W.sum(axis=1, keepdims=True)
        return np.where(S > 0, W / np.where(S > 0, S, 1), np.nan)
    Xl = U[:, :, None] ** np.arange(degre + 1)
    A = np.einsum("gnj,gn,gnk->gjk", Xl, W, Xl) + 1e-12 * np.eye(degre + 1)
    L = np.linalg.solve(A, Xl.transpose(0, 2, 1) * W[:, None, :])[:, 0, :]
    return np.where(W.sum(axis=1, keepdims=True) > 0, L, np.nan)


# Taille de bloc pour que les tableaux (bloc, n, degre + 1) restent raisonnables.
def bloc(n, degre):
    return max(1, 3_000_000 // (n * (degre + 1)))


# Estimation en chaque point de la grille (par blocs). Avec poids=True, renvoie aussi L.
def lisser(x, y, grille, h, noyau="gaussien", degre=1, poids=False):
    x, y, grille = (np.asarray(a, dtype=float) for a in (x, y, grille))
    fit, L = [], []
    for d in range(0, len(grille), bloc(len(x), degre)):
        Lb = poids_locaux(x, grille[d:d + bloc(len(x), degre)], h, noyau, degre)
        fit.append(Lb @ y)
        if poids:
            L.append(Lb)
    fit = np.concatenate(fit)
    return (fit, np.vstack(L)) if poids else fit


# Erreur de validation croisee leave-one-out. Pour un lisseur lineaire de ce type, le residu
# obtenu en retirant l'observation i vaut e_i / (1 - L_ii), sans refaire n estimations.
def cv_loo(x, y, h, noyau="gaussien", degre=1):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    erreurs = np.empty(n)
    for d in range(0, n, bloc(n, degre)):
        idx = np.arange(d, min(d + bloc(n, degre), n))
        L = poids_locaux(x, x[idx], h, noyau, degre)
        erreurs[idx] = (y[idx] - L @ y) / (1 - L[np.arange(len(idx)), idx])
    return np.mean(erreurs ** 2)


# Fenetre qui minimise la validation croisee : recherche sur une grille geometrique
# puis affinage autour du meilleur point.
def choisir_h(x, y, noyau="gaussien", degre=1, hs=None):
    x = np.asarray(x, dtype=float)
    ecart = np.std(x)
    hs = ecart * np.geomspace(0.02, 1.5, 25) if hs is None else np.asarray(hs, dtype=float)
    scores = np.array([cv_loo(x, y, h, noyau, degre) for h in hs])
    i = int(np.nanargmin(scores))
    bas, haut = hs[max(i - 1, 0)], hs[min(i + 1, len(hs) - 1)]
    fin = minimize_scalar(lambda h: cv_loo(x, y, h, noyau, degre), bounds=(bas, haut), method="bounded",
                          options={"xatol": 1e-6 * ecart})
    return fin.x


# Estimation avec intervalles de confiance ponctuels (variance robuste a l'heteroscedasticite :
# var = somme l_i^2 e_i^2). Les intervalles ignorent le biais de lissage.
def intervalle(x, y, grille, h, noyau="gaussien", degre=1, niveau=0.95):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    e = y - lisser(x, y, x, h, noyau, degre)
    fit, L = lisser(x, y, grille, h, noyau, degre, poids=True)
    se = np.sqrt((L ** 2) @ (e ** 2))
    z = norm.ppf(0.5 + niveau / 2)
    return fit, fit - z * se, fit + z * se
