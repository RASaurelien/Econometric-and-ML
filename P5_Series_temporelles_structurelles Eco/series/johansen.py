import numpy as np
from scipy import linalg

from .var import ResultatVAR

# Les cas traites : "constante" (constante libre dans le VECM, donc derive dans les niveaux)
# ou "aucune" (pas de terme deterministe).


# Residus d'une regression de Z sur W (plusieurs series a la fois si Z et W ont des dimensions de tete).
def residus(Z, W):
    if W.shape[-1] == 0:
        return Z
    coef = np.linalg.solve(np.einsum("...ni,...nj->...ij", W, W), np.einsum("...ni,...nj->...ij", W, Z))
    return Z - W @ coef


# R0 (variations) et R1 (niveaux retardes) apres retrait des differences retardees et de la constante.
# Un VAR(p) en niveaux correspond a p - 1 differences retardees.
def preparer(Y, p, constante):
    dY = np.diff(Y, axis=0)
    n = len(Y) - p
    Z0, Z1 = dY[p - 1:], Y[p - 1:-1]
    W = [dY[p - 1 - j:len(dY) - j] for j in range(1, p)]
    if constante:
        W.append(np.ones((n, 1)))
    W = np.hstack(W) if W else np.empty((n, 0))
    return residus(Z0, W), residus(Z1, W), W, Z0, Z1


# Matrices de moments S00, S01, S11 (divisees par n).
def moments(R0, R1):
    n = R0.shape[-2]
    return (np.einsum("...ni,...nj->...ij", R0, R0) / n, np.einsum("...ni,...nj->...ij", R0, R1) / n,
            np.einsum("...ni,...nj->...ij", R1, R1) / n)


# Valeurs propres de S11^-1 S10 S00^-1 S01, par ordre decroissant, dans [0, 1).
def valeurs_propres(S00, S01, S11):
    M = np.linalg.solve(S11, np.swapaxes(S01, -1, -2) @ np.linalg.solve(S00, S01))
    lam = np.sort(np.linalg.eigvals(M).real, axis=-1)[..., ::-1]
    return np.clip(lam, 0, 1 - 1e-12)


# Statistiques de la trace et de la valeur propre maximale, pour r = 0, ..., k-1.
def statistiques(lam, n):
    log = -n * np.log(1 - lam)
    return np.cumsum(log[..., ::-1], axis=-1)[..., ::-1], log


_cache = {}


# Valeurs critiques simulees pour d = k - r tendances communes (loi asymptotique sous H0 : r = 0),
# obtenues en appliquant la procedure a des marches aleatoires de dimension d. Avec constante libre,
# les marches ont une derive non nulle (cas habituel des tables de Johansen) : une des tendances
# devient alors deterministe, ce qui change la loi, par exemple chi2(1) pour d = 1.
# Renvoie {"trace": {niveau: valeur}, "maxeig": {niveau: valeur}} pour 0.90, 0.95, 0.99.
def valeurs_critiques(d, constante=True, reps=10000, T=500, graine=0):
    cle = (d, constante, reps, T, graine)
    if cle not in _cache:
        rng = np.random.default_rng(graine)
        trace, maxe = [], []
        for _ in range(reps // 1000):
            W = np.cumsum(rng.normal(size=(1000, T + 1, d)) + float(constante), axis=1)
            Z0, Z1 = np.diff(W, axis=1), W[:, :-1]
            if constante:
                Z0, Z1 = Z0 - Z0.mean(axis=1, keepdims=True), Z1 - Z1.mean(axis=1, keepdims=True)
            tr, mx = statistiques(valeurs_propres(*moments(Z0, Z1)), T)
            trace.append(tr[:, 0])
            maxe.append(mx[:, 0])
        niveaux = [0.90, 0.95, 0.99]
        _cache[cle] = {nom: dict(zip(niveaux, np.quantile(np.concatenate(v), niveaux)))
                       for nom, v in (("trace", trace), ("maxeig", maxe))}
    return _cache[cle]


# Resultat du test de Johansen.
class ResultatJohansen:
    def __init__(self, lam, trace, maxeig, n, constante):
        self.valeurs_propres, self.trace, self.maxeig = lam, trace, maxeig
        self.n, self.constante, self.k = n, constante, len(lam)

    # Rang retenu : la premiere hypothese r = 0, 1, ... non rejetee au niveau donne.
    def rang(self, niveau=0.95, test="trace"):
        stats_ = self.trace if test == "trace" else self.maxeig
        for r in range(self.k):
            if stats_[r] < valeurs_critiques(self.k - r, self.constante)[test][niveau]:
                return r
        return self.k

    def __str__(self):
        lignes = [f"{'H0 : r =':>9}{'valeur propre':>15}{'trace':>10}{'cv 95 %':>10}{'max eig':>10}{'cv 95 %':>10}"]
        for r in range(self.k):
            cv = valeurs_critiques(self.k - r, self.constante)
            lignes.append(f"{r:>9}{self.valeurs_propres[r]:>15.4f}{self.trace[r]:>10.2f}{cv['trace'][0.95]:>10.2f}"
                          f"{self.maxeig[r]:>10.2f}{cv['maxeig'][0.95]:>10.2f}")
        return "\n".join(lignes)


# Test de Johansen sur Y (T, k), pour un VAR(p) en niveaux.
def johansen(Y, p=2, deterministe="constante"):
    constante = deterministe == "constante"
    R0, R1, _, _, _ = preparer(Y, p, constante)
    n = len(R0)
    lam = valeurs_propres(*moments(R0, R1))
    trace, maxeig = statistiques(lam, n)
    return ResultatJohansen(lam, trace, maxeig, n, constante)


# Modele a correction d'erreur de rang r :
#   dY_t = alpha beta' Y_{t-1} + Gamma_1 dY_{t-1} + ... + Gamma_{p-1} dY_{t-p+1} + mu + e_t.
# beta est normalise pour que son bloc superieur r x r soit l'identite.
class ResultatVECM:
    def __init__(self, Y, p, r, constante):
        R0, R1, W, Z0, Z1 = preparer(Y, p, constante)
        S00, S01, S11 = moments(R0, R1)
        k, n = Y.shape[1], len(R0)
        self.Y, self.p, self.r, self.constante, self.n, self.k = Y, p, r, constante, n, k
        self.valeurs_propres, V = linalg.eigh(S01.T @ np.linalg.solve(S00, S01), S11)
        self.valeurs_propres, V = self.valeurs_propres[::-1], V[:, ::-1]
        beta = V[:, :r] @ np.linalg.inv(V[:r, :r])
        self.beta = beta
        self.alpha = S01 @ beta @ np.linalg.inv(beta.T @ S11 @ beta)
        self.Pi = self.alpha @ beta.T
        reste = Z0 - Z1 @ self.Pi.T
        coef = np.linalg.solve(W.T @ W, W.T @ reste) if W.shape[1] else np.empty((0, k))
        self.resid = reste - W @ coef
        self.sigma_ml = self.resid.T @ self.resid / n
        self.mu = coef[-1] if constante else np.zeros(k)
        self.Gamma = np.stack([coef[j * k:(j + 1) * k].T for j in range(p - 1)]) if p > 1 else np.empty((0, k, k))

    # Ecriture du meme modele comme VAR en niveaux : A_1 = I + Pi + Gamma_1, A_i = Gamma_i - Gamma_{i-1}.
    def en_var(self):
        k, p = self.k, self.p
        G = self.Gamma
        A = [np.eye(k) + self.Pi + (G[0] if p > 1 else 0)]
        for i in range(1, p - 1):
            A.append(G[i] - G[i - 1])
        if p > 1:
            A.append(-G[-1])
        return np.stack(A), self.mu


def vecm(Y, p=2, r=1, deterministe="constante"):
    return ResultatVECM(np.asarray(Y, dtype=float), p, r, deterministe == "constante")
