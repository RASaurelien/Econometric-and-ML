import numpy as np

from .var import ResultatVAR


# Coefficients MA du VAR : Phi_0 = I, Phi_h = somme_j A_j Phi_{h-j}.
# A a la forme (..., p, k, k), le resultat (..., n_ahead + 1, k, k).
def coefs_ma(A, n_ahead):
    p, k = A.shape[-3], A.shape[-1]
    Phi = [np.broadcast_to(np.eye(k), A.shape[:-3] + (k, k))]
    for h in range(1, n_ahead + 1):
        Phi.append(sum(A[..., j, :, :] @ Phi[h - 1 - j] for j in range(min(p, h))))
    return np.stack(Phi, axis=-3)


# Effet de long terme des chocs de forme reduite : C(1) = (I - A_1 - ... - A_p)^-1.
def effet_long_terme(A):
    return np.linalg.inv(np.eye(A.shape[-1]) - A.sum(axis=-3))


# Matrice d'impact B0 telle que B0 B0' = sigma, avec le choc j qui n'agit que
# sur les variables j, j+1, ... a l'impact : identification de Cholesky (ordre des variables).
def impact_cholesky(A, sigma):
    return np.linalg.cholesky(sigma)


# Identification de Blanchard-Quah : l'effet cumule C(1) B0 est triangulaire inferieur,
# donc le choc j n'a aucun effet de long terme sur les variables 1 a j-1.
# La diagonale de l'effet de long terme est positive.
def impact_long_terme(A, sigma):
    C1 = effet_long_terme(A)
    L = np.linalg.cholesky(C1 @ sigma @ np.swapaxes(C1, -1, -2))
    return np.linalg.inv(C1) @ L


IMPACTS = {"cholesky": impact_cholesky, "long_terme": impact_long_terme}


# Fonctions de reponse impulsionnelle : [h, i, j] = effet du choc j sur la variable i a l'horizon h.
def irf(A, impact, n_ahead):
    return coefs_ma(A, n_ahead) @ impact[..., None, :, :]


# Decomposition de variance : part de la variance d'erreur de prevision de la variable i
# a l'horizon h due au choc j. Les parts de chaque ligne somment a 1.
def fevd(A, impact, n_ahead):
    theta2 = irf(A, impact, n_ahead - 1) ** 2
    cumul = np.cumsum(theta2, axis=-3)
    return cumul / cumul.sum(axis=-1, keepdims=True)


# MCO d'un VAR(p) sur de nombreuses series a la fois, Ys de forme (B, T, k).
# Renvoie A (B, p, k, k), la constante (B, k) et sigma corrigee (B, k, k).
def var_batch(Ys, p, constante=True):
    B, T, k = Ys.shape
    X = np.concatenate([Ys[:, p - j:T - j] for j in range(1, p + 1)], axis=2)
    if constante:
        X = np.concatenate([np.ones((B, T - p, 1)), X], axis=2)
    y = Ys[:, p:]
    params = np.linalg.solve(np.einsum("bnr,bns->brs", X, X), np.einsum("bnr,bnk->brk", X, y))
    resid = y - X @ params
    sigma = np.einsum("bni,bnj->bij", resid, resid) / (T - p - X.shape[2])
    deca = int(constante)
    A = np.stack([np.swapaxes(params[:, deca + j * k:deca + (j + 1) * k], 1, 2) for j in range(p)], axis=1)
    c = params[:, 0] if constante else np.zeros((B, k))
    return A, c, sigma


# Intervalles de confiance par bootstrap sur les residus (percentile). On recentre les residus,
# on les retire avec remise, on regenere B series a partir des p premieres observations,
# on reestime le VAR et on recalcule les IRF. Renvoie (borne basse, borne haute).
def irf_bootstrap(res, n_ahead, structure="cholesky", B=500, niveau=0.95, graine=None):
    rng = np.random.default_rng(graine)
    k, p, T = res.k, res.p, len(res.Y)
    e = res.resid - res.resid.mean(axis=0)
    tirages = e[rng.integers(0, len(e), size=(B, T - p))]
    Ys = np.empty((B, T, k))
    Ys[:, :p] = res.Y[:p]
    for t in range(p, T):
        Ys[:, t] = res.c + sum(Ys[:, t - 1 - j] @ res.A[j].T for j in range(p)) + tirages[:, t - p]
    A, _, sigma = var_batch(Ys, p, res.constante)
    reponses = irf(A, IMPACTS[structure](A, sigma), n_ahead)
    q = (1 - niveau) / 2
    return np.quantile(reponses, q, axis=0), np.quantile(reponses, 1 - q, axis=0)


# Raccourci : IRF d'un VAR estime, avec la structure choisie.
def irf_var(res, n_ahead, structure="cholesky"):
    return irf(res.A, IMPACTS[structure](res.A, res.sigma), n_ahead)


def fevd_var(res, n_ahead, structure="cholesky"):
    return fevd(res.A, IMPACTS[structure](res.A, res.sigma), n_ahead)
