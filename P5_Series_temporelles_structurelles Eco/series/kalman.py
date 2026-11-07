import ctypes
import os
import shutil

import numpy as np
from scipy import linalg, optimize, stats

if os.name == "nt":
    _compilateur = shutil.which("g++")
    _dossier_dll = os.add_dll_directory(os.path.dirname(_compilateur)) if _compilateur else None
    NOM_BIBLIOTHEQUE = "libkalman.dll"
else:
    _dossier_dll = None
    NOM_BIBLIOTHEQUE = "libkalman.so"

CHEMIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), NOM_BIBLIOTHEQUE)


# Charge la bibliotheque C++ si elle a ete compilee, sinon renvoie None.
def charger():
    try:
        lib = ctypes.CDLL(CHEMIN)
    except OSError:
        return None
    lib.kalman_filtre.restype = ctypes.c_int
    return lib


LIB = charger()


def cpp_disponible():
    return LIB is not None


def pointeur(a):
    return None if a is None else a.ctypes.data_as(ctypes.c_void_p)


# Resultat du filtrage. a_pred et P_pred ont n+1 lignes (la derniere sert a prevoir).
class Filtrage:
    def __init__(self, loglik, a_pred=None, P_pred=None, a_filt=None, P_filt=None):
        self.loglik = loglik
        self.a_pred, self.P_pred = a_pred, P_pred
        self.a_filt, self.P_filt = a_filt, P_filt


# Etat initial stationnaire : P0 resout P = T P T' + R Q R'.
def init_stationnaire(T, RQR):
    return linalg.solve_discrete_lyapunov(T, RQR)


# Initialisation diffuse approchee : variance tres grande.
def init_diffuse(m, kappa=1e7):
    return kappa * np.eye(m)


# Meme algorithme que le C++, en numpy. Sert de reference et de secours.
def filtre_numpy(y, Z, d, H, T, c, RQR, a0, P0, stocker):
    n, q = y.shape
    m = T.shape[0]
    a, P, ll = a0.copy(), P0.copy(), 0.0
    sorties = [np.empty((n + 1, m)), np.empty((n + 1, m, m)), np.empty((n, m)), np.empty((n, m, m))]
    for t in range(n):
        sorties[0][t], sorties[1][t] = a, P
        obs = ~np.isnan(y[t])
        if obs.any():
            Zs = Z[obs]
            v = y[t, obs] - d[obs] - Zs @ a
            PZt = P @ Zs.T
            F = Zs @ PZt + H[np.ix_(obs, obs)]
            try:
                L = np.linalg.cholesky(F)
            except np.linalg.LinAlgError:
                return None
            Finv = np.linalg.inv(F)
            K = PZt @ Finv
            af = a + K @ v
            Pf = P - K @ PZt.T
            Pf = 0.5 * (Pf + Pf.T)
            ll += -0.5 * (obs.sum() * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + v @ Finv @ v)
        else:
            af, Pf = a, P
        sorties[2][t], sorties[3][t] = af, Pf
        a = T @ af + c
        P = T @ Pf @ T.T + RQR
        P = 0.5 * (P + P.T)
    sorties[0][n], sorties[1][n] = a, P
    return Filtrage(ll, *sorties) if stocker else Filtrage(ll)


def tableau2(x, forme=None):
    x = np.ascontiguousarray(x, dtype=float)
    return x if forme is None else x.reshape(forme)


# Filtre de Kalman. y (n, q) avec NaN pour les valeurs manquantes.
# Z (q, m), H (q, q), T (m, m), Q (r, r), R (m, r) = identite par defaut.
# Sans P0, l'etat initial est la loi stationnaire (T doit etre stable), de moyenne (I - T)^-1 c.
# moteur : "auto" (C++ si disponible), "cpp" ou "numpy".
def filtre(y, Z, H, T, Q, R=None, d=None, c=None, a0=None, P0=None, stocker=True, moteur="auto"):
    y = np.asarray(y, dtype=float)
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    n, q = y.shape
    T = np.atleast_2d(np.asarray(T, dtype=float))
    m = T.shape[0]
    Z = np.atleast_2d(np.asarray(Z, dtype=float)).reshape(q, m)
    H = np.atleast_2d(np.asarray(H, dtype=float)).reshape(q, q)
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.eye(m) if R is None else np.asarray(R, dtype=float).reshape(m, -1)
    RQR = R @ Q @ R.T
    d = np.zeros(q) if d is None else np.asarray(d, dtype=float).reshape(q)
    c = np.zeros(m) if c is None else np.asarray(c, dtype=float).reshape(m)
    if P0 is None:
        P0 = init_stationnaire(T, RQR)
        a0 = np.linalg.solve(np.eye(m) - T, c) if a0 is None else np.asarray(a0, dtype=float).reshape(m)
    else:
        P0 = np.asarray(P0, dtype=float).reshape(m, m)
        a0 = np.zeros(m) if a0 is None else np.asarray(a0, dtype=float).reshape(m)

    if moteur == "cpp" and LIB is None:
        raise RuntimeError("bibliotheque C++ absente, lancer make")
    if moteur == "numpy" or LIB is None:
        res = filtre_numpy(y, Z, d, H, T, c, RQR, a0, P0, stocker)
    else:
        entrees = [tableau2(x) for x in (y, Z, d, H, T, c, RQR, a0, P0)]
        ll = np.zeros(1)
        sorties = [np.empty((n + 1, m)), np.empty((n + 1, m, m)), np.empty((n, m)), np.empty((n, m, m))]
        ptrs = [pointeur(s) if stocker else None for s in sorties]
        code = LIB.kalman_filtre(n, q, m, *[pointeur(x) for x in entrees], *ptrs, pointeur(ll))
        res = None if code else (Filtrage(ll[0], *sorties) if stocker else Filtrage(ll[0]))
    if res is None:
        res = Filtrage(-np.inf)
    return res


# Lisseur de Rauch-Tung-Striebel, a partir d'un filtrage deja calcule.
def lisseur(f, T):
    T = np.atleast_2d(np.asarray(T, dtype=float))
    a_s, P_s = f.a_filt.copy(), f.P_filt.copy()
    for t in range(len(a_s) - 2, -1, -1):
        J = f.P_filt[t] @ T.T @ np.linalg.pinv(f.P_pred[t + 1])
        a_s[t] = f.a_filt[t] + J @ (a_s[t + 1] - f.a_pred[t + 1])
        P_s[t] = f.P_filt[t] + J @ (P_s[t + 1] - f.P_pred[t + 1]) @ J.T
    return a_s, P_s


# Derivees numeriques par differences centrees.
def jacobien(f, x, h=1e-6):
    cols = []
    for j in range(len(x)):
        pas = h * max(1.0, abs(x[j]))
        e = np.zeros(len(x))
        e[j] = pas
        cols.append((np.atleast_1d(f(x + e)) - np.atleast_1d(f(x - e))) / (2 * pas))
    return np.column_stack(cols)


def hessien_num(f, x, h=1e-4):
    p = len(x)
    pas = h * np.maximum(1.0, np.abs(x))
    H = np.zeros((p, p))
    for j in range(p):
        for k in range(j, p):
            ej, ek = np.zeros(p), np.zeros(p)
            ej[j], ek[k] = pas[j], pas[k]
            H[j, k] = H[k, j] = (f(x + ej + ek) - f(x + ej - ek) - f(x - ej + ek) + f(x - ej - ek)) / (
                4 * pas[j] * pas[k])
    return H


# Resultat d'un maximum de vraisemblance sur un modele d'etat.
class ResultatEtat:
    def __init__(self, theta, v, noms, loglik, n, y, matrices, options):
        self.theta, self.cov, self.noms, self.loglik, self.n = theta, v, noms, loglik, n
        self.y, self.matrices, self.options = y, matrices, options
        self.k = len(theta)
        self.se = np.sqrt(np.diag(v))
        self.z = theta / self.se
        self.p = 2 * stats.norm.sf(np.abs(self.z))
        self.aic = 2 * self.k - 2 * loglik
        self.bic = self.k * np.log(n) - 2 * loglik

    def filtre(self):
        return filtre(self.y, **self.matrices, **self.options)

    def lisse(self):
        return lisseur(self.filtre(), self.matrices["T"])

    def __str__(self):
        lignes = [f"{'':>12}{'coef':>12}{'se':>12}{'z':>10}{'p':>10}"]
        for i, nom in enumerate(self.noms):
            lignes.append(f"{nom:>12}{self.theta[i]:>12.4f}{self.se[i]:>12.4f}{self.z[i]:>10.2f}{self.p[i]:>10.3f}")
        lignes.append(f"n = {self.n}, log-vraisemblance = {self.loglik:.3f}, AIC = {self.aic:.1f}, BIC = {self.bic:.1f}")
        return "\n".join(lignes)


# Maximum de vraisemblance par BFGS. construire(theta) renvoie les matrices du modele
# (arguments de filtre), transformation passe des parametres libres u aux parametres
# theta (par exemple exp pour une variance). Les ecarts-types suivent par delta method.
def estimer(y, construire, u0, transformation, noms, options=None):
    options = {} if options is None else options
    y = np.asarray(y, dtype=float)
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    def objectif(u):
        try:
            f = filtre(y, **construire(transformation(u)), stocker=False, **options)
        except (np.linalg.LinAlgError, ValueError):
            return 1e10
        return 1e10 if not np.isfinite(f.loglik) else -f.loglik

    res = optimize.minimize(objectif, u0, method="BFGS", options={"gtol": 1e-8})
    u = res.x
    v = np.linalg.inv(hessien_num(objectif, u))
    J = jacobien(transformation, u)
    theta = transformation(u)
    matrices = construire(theta)
    ll = filtre(y, **matrices, stocker=False, **options).loglik
    return ResultatEtat(theta, J @ v @ J.T, noms, ll, int(np.sum(~np.isnan(y))), y, matrices, options)


# Modele de niveau local : y_t = mu_t + eps_t, mu_{t+1} = mu_t + eta_t.
# Parametres : variance de eps et variance de eta. Initialisation diffuse.
def niveau_local(y, moteur="auto"):
    def construire(theta):
        return dict(Z=[[1.0]], H=[[theta[0]]], T=[[1.0]], Q=[[theta[1]]], P0=init_diffuse(1))

    y = np.asarray(y, dtype=float)
    u0 = np.log([np.nanvar(np.diff(y)) / 2] * 2)
    return estimer(y, construire, u0, np.exp, ["var_eps", "var_eta"], {"moteur": moteur})


# ARMA(p, q) avec moyenne, vraisemblance exacte par filtre de Kalman (forme de Harvey).
# Parametres : moyenne, phi_1..phi_p, theta_1..theta_q, variance de l'innovation.
def arma(y, p, q, moteur="auto"):
    r = max(p, q + 1)

    def construire(theta):
        mu, phi, psi, sigma2 = theta[0], theta[1:1 + p], theta[1 + p:1 + p + q], theta[-1]
        T = np.zeros((r, r))
        T[:p, 0] = phi
        T[:-1, 1:] += np.eye(r - 1)
        if p and np.max(np.abs(np.linalg.eigvals(T))) >= 0.999:
            raise ValueError("non stationnaire")
        R = np.zeros((r, 1))
        R[0, 0] = 1
        R[1:1 + q, 0] = psi
        Z = np.zeros((1, r))
        Z[0, 0] = 1
        return dict(Z=Z, H=[[0.0]], T=T, Q=[[sigma2]], R=R, d=[mu])

    y = np.asarray(y, dtype=float)
    u0 = np.concatenate([[np.nanmean(y)], np.zeros(p + q), [np.log(np.nanvar(y))]])
    transformation = lambda u: np.concatenate([u[:-1], [np.exp(u[-1])]])
    noms = ["const"] + [f"ar{i}" for i in range(1, p + 1)] + [f"ma{i}" for i in range(1, q + 1)] + ["sigma2"]
    return estimer(y, construire, u0, transformation, noms, {"moteur": moteur})
