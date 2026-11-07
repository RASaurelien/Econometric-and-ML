import numpy as np
from scipy import stats


# Variables dependantes et regresseurs d'un VAR(p). depart permet d'utiliser le meme
# echantillon pour comparer plusieurs ordres.
def regresseurs(Y, p, constante, depart):
    T = len(Y)
    X = np.hstack([Y[depart - j:T - j] for j in range(1, p + 1)])
    if constante:
        X = np.column_stack([np.ones(len(X)), X])
    return Y[depart:], X


# VAR(p) estime par MCO, equation par equation (memes regresseurs partout).
# params a la forme (regresseurs, equations) : constante d'abord, puis retard 1, retard 2, etc.
# A[j] est la matrice du retard j+1, sigma est corrigee des degres de liberte.
class ResultatVAR:
    def __init__(self, Y, p, constante=True, depart=None):
        depart = p if depart is None else depart
        y, X = regresseurs(Y, p, constante, depart)
        self.Y, self.p, self.constante = Y, p, constante
        self.n, self.k = y.shape
        self.n_reg = X.shape[1]
        self.df_resid = self.n - self.n_reg
        self.XtX_inv = np.linalg.inv(X.T @ X)
        self.params = self.XtX_inv @ X.T @ y
        self.resid = y - X @ self.params
        self.sigma_ml = self.resid.T @ self.resid / self.n
        self.sigma = self.resid.T @ self.resid / self.df_resid
        k, deca = self.k, int(constante)
        self.c = self.params[0] if constante else np.zeros(k)
        self.A = np.stack([self.params[deca + j * k:deca + (j + 1) * k].T for j in range(p)])
        self.se = np.sqrt(np.outer(np.diag(self.XtX_inv), np.diag(self.sigma)))
        self.t = self.params / self.se
        self.p_valeurs = 2 * stats.t.sf(np.abs(self.t), self.df_resid)

    # Matrice compagnon, de taille kp x kp.
    def compagnon(self):
        k, p = self.k, self.p
        M = np.zeros((k * p, k * p))
        M[:k] = np.hstack(list(self.A))
        M[k:, :-k] = np.eye(k * (p - 1))
        return M

    # Le VAR est stable si toutes les racines de la matrice compagnon sont dans le cercle unite.
    def stable(self):
        return bool(np.max(np.abs(np.linalg.eigvals(self.compagnon()))) < 1)

    # Previsions ponctuelles a h pas.
    def prevoir(self, h):
        histoire = list(self.Y[-self.p:])
        sortie = []
        for _ in range(h):
            y = self.c + sum(self.A[j] @ histoire[-1 - j] for j in range(self.p))
            histoire.append(y)
            sortie.append(y)
        return np.array(sortie)

    # Causalite de Granger : H0 = les retards des variables cause n'entrent dans aucune
    # des equations des variables effet. Test de Wald divise par le nombre de restrictions,
    # loi F(q, k * ddl residuel) comme dans statsmodels.
    def granger(self, cause, effet):
        cause, effet = np.atleast_1d(cause), np.atleast_1d(effet)
        deca, k = int(self.constante), self.k
        lignes = [deca + j * k + i for j in range(self.p) for i in cause]
        positions = [(r, e) for r in lignes for e in effet]
        b = np.array([self.params[r, e] for r, e in positions])
        V = np.array([[self.sigma[e1, e2] * self.XtX_inv[r1, r2] for r2, e2 in positions]
                      for r1, e1 in positions])
        wald = b @ np.linalg.solve(V, b)
        q = len(positions)
        stat = wald / q
        return stat, stats.f.sf(stat, q, k * self.df_resid)


# Criteres d'information pour les ordres 1 a pmax, sur le meme echantillon.
# Renvoie un dictionnaire de tableaux (un element par ordre) et l'ordre choisi par critere.
def choisir_ordre(Y, pmax, constante=True):
    ics = {"aic": [], "bic": [], "hqic": [], "fpe": []}
    for p in range(1, pmax + 1):
        r = ResultatVAR(Y, p, constante, depart=pmax)
        n, k = r.n, r.k
        ld = np.log(np.linalg.det(r.sigma_ml))
        libres = p * k * k + k * int(constante)
        ics["aic"].append(ld + 2 / n * libres)
        ics["bic"].append(ld + np.log(n) / n * libres)
        ics["hqic"].append(ld + 2 * np.log(np.log(n)) / n * libres)
        ics["fpe"].append(((n + r.n_reg) / r.df_resid) ** k * np.exp(ld))
    ics = {nom: np.array(v) for nom, v in ics.items()}
    return ics, {nom: int(v.argmin()) + 1 for nom, v in ics.items()}
