import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm

from .estimation import estimer


# Ajoute la constante et fabrique les noms des colonnes.
def donnees(X, constante, noms):
    X = np.asarray(X, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    noms = list(noms) if noms is not None else [f"x{i}" for i in range(1, X.shape[1] + 1)]
    if constante:
        X = np.column_stack([np.ones(len(X)), X])
        noms = ["const"] + noms
    return X, noms


# Logit : log-vraisemblance, scores et hessien analytiques.
def ll_logit(theta, y, X):
    return -np.logaddexp(0, -(2 * y - 1) * (X @ theta))


def scores_logit(theta, y, X):
    return (y - expit(X @ theta))[:, None] * X


def hessien_logit(theta, y, X):
    p = expit(X @ theta)
    return -(X * (p * (1 - p))[:, None]).T @ X


# Probit : lambda est le score par rapport a x'b.
def lambda_probit(theta, y, X):
    q = 2 * y - 1
    z = q * (X @ theta)
    return q * np.exp(norm.logpdf(z) - norm.logcdf(z))


def ll_probit(theta, y, X):
    return norm.logcdf((2 * y - 1) * (X @ theta))


def scores_probit(theta, y, X):
    return lambda_probit(theta, y, X)[:, None] * X


def hessien_probit(theta, y, X):
    lam = lambda_probit(theta, y, X)
    return -(X * (lam * (lam + X @ theta))[:, None]).T @ X


LIENS = {
    "logit": (ll_logit, scores_logit, hessien_logit, expit),
    "probit": (ll_probit, scores_probit, hessien_probit, norm.cdf),
}


# Modele binaire generique. La moyenne conditionnelle est P(y = 1 | x).
def modele_binaire(lien, y, X, constante, cov, groupes, noms, methode):
    ll, sc, he, F = LIENS[lien]
    y = np.asarray(y, dtype=float)
    X, noms = donnees(X, constante, noms)
    ll_nul = None
    if constante:
        m = y.mean()
        ll_nul = len(y) * (m * np.log(m) + (1 - m) * np.log(1 - m))
    res = estimer(
        lambda t: ll(t, y, X), np.zeros(X.shape[1]), noms, cov, groupes, methode,
        scores=lambda t: sc(t, y, X), hessien=lambda t: he(t, y, X), ll_nul=ll_nul,
    )
    res.moyenne = lambda theta, X_: F(X_ @ theta)
    res.X, res.noms_x = X, noms
    return res


def logit(y, X, constante=True, cov="hessien", groupes=None, noms=None, methode="newton"):
    return modele_binaire("logit", y, X, constante, cov, groupes, noms, methode)


def probit(y, X, constante=True, cov="hessien", groupes=None, noms=None, methode="newton"):
    return modele_binaire("probit", y, X, constante, cov, groupes, noms, methode)


# Logit ou probit ordonne. Pas de constante : elle est absorbee par les seuils.
# On estime le premier seuil et le log des ecarts entre seuils pour garder
# l'ordre, puis on affiche les seuils eux-memes.
def ordonne(y, X, lien="logit", cov="hessien", groupes=None, noms=None, methode="bfgs"):
    F, Finv = (expit, lambda p: np.log(p / (1 - p))) if lien == "logit" else (norm.cdf, norm.ppf)
    X, noms = donnees(X, False, noms)
    cats, codes = np.unique(y, return_inverse=True)
    K, J, n = X.shape[1], len(cats), len(codes)

    def seuils(theta):
        return np.concatenate([[theta[K]], theta[K] + np.cumsum(np.exp(theta[K + 1:]))])

    def ll(theta):
        mu = np.concatenate([[-np.inf], seuils(theta), [np.inf]])
        xb = X @ theta[:K]
        p = F(mu[codes + 1] - xb) - F(mu[codes] - xb)
        return np.log(np.maximum(p, 1e-300))

    depart = Finv(np.cumsum(np.bincount(codes))[:-1] / n)
    theta0 = np.concatenate([np.zeros(K), [depart[0]], np.log(np.maximum(np.diff(depart), 0.1))])
    noms_seuils = [f"seuil {cats[i]}|{cats[i + 1]}" for i in range(J - 1)]
    res = estimer(
        ll, theta0, noms + noms_seuils, cov, groupes, methode,
        transformation=lambda t: np.concatenate([t[:K], seuils(t)]),
    )

    def moyenne(theta, X_, categorie):
        j = int(np.searchsorted(cats, categorie))
        mu = np.concatenate([[-np.inf], theta[K:], [np.inf]])
        xb = X_ @ theta[:K]
        return F(mu[j + 1] - xb) - F(mu[j] - xb)

    res.moyenne, res.categories, res.X, res.noms_x = moyenne, cats, X, noms
    return res


# Logit multinomial. La premiere categorie sert de reference.
def multinomial(y, X, constante=True, cov="hessien", groupes=None, noms=None, methode="bfgs"):
    X, noms = donnees(X, constante, noms)
    cats, codes = np.unique(y, return_inverse=True)
    K, J, n = X.shape[1], len(cats), len(codes)

    def utilites(theta, X_):
        B = theta.reshape(J - 1, K)
        return np.column_stack([np.zeros(len(X_)), X_ @ B.T])

    def ll(theta):
        U = utilites(theta, X)
        return U[np.arange(n), codes] - logsumexp(U, axis=1)

    noms_all = [f"{c}:{nom}" for c in cats[1:] for nom in noms]
    res = estimer(ll, np.zeros((J - 1) * K), noms_all, cov, groupes, methode)

    def moyenne(theta, X_, categorie):
        U = utilites(theta, X_)
        return np.exp(U[:, int(np.searchsorted(cats, categorie))] - logsumexp(U, axis=1))

    res.moyenne, res.categories, res.X, res.noms_x = moyenne, cats, X, noms
    return res


# Tobit, avec censure a gauche et/ou a droite. On estime log(sigma), on affiche sigma.
# moyenne(quoi="observee") donne E[y | x], moyenne(quoi="latente") donne x'b.
def tobit(y, X, constante=True, gauche=0.0, droite=None, cov="hessien", groupes=None,
          noms=None, methode="bfgs"):
    y = np.asarray(y, dtype=float)
    X, noms = donnees(X, constante, noms)
    K = X.shape[1]
    cg = y <= gauche if gauche is not None else np.zeros(len(y), dtype=bool)
    cd = y >= droite if droite is not None else np.zeros(len(y), dtype=bool)
    libre = ~cg & ~cd

    def ll(theta):
        xb, s = X @ theta[:K], np.exp(theta[K])
        out = np.zeros(len(y))
        if gauche is not None:
            out[cg] = norm.logcdf((gauche - xb[cg]) / s)
        if droite is not None:
            out[cd] = norm.logsf((droite - xb[cd]) / s)
        out[libre] = norm.logpdf((y[libre] - xb[libre]) / s) - np.log(s)
        return out

    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    theta0 = np.concatenate([b0, [np.log(np.std(y - X @ b0))]])
    res = estimer(
        ll, theta0, noms + ["sigma"], cov, groupes, methode,
        transformation=lambda t: np.concatenate([t[:K], [np.exp(t[K])]]),
    )

    def moyenne(theta, X_, quoi="observee"):
        xb, s = X_ @ theta[:K], theta[K]
        if quoi == "latente":
            return xb
        a = (gauche - xb) / s if gauche is not None else -np.inf
        b = (droite - xb) / s if droite is not None else np.inf
        m = (norm.cdf(b) - norm.cdf(a)) * xb + s * (norm.pdf(a) - norm.pdf(b))
        if gauche is not None:
            m = m + gauche * norm.cdf(a)
        if droite is not None:
            m = m + droite * norm.sf(b)
        return m

    res.moyenne, res.X, res.noms_x = moyenne, X, noms
    return res


# Modele de selection de Heckman estime par maximum de vraisemblance.
# y n'est lu que pour les observations selectionnees (s = 1). Z doit contenir
# au moins une variable absente de X pour bien identifier le modele.
def heckman(y, X, s, Z, constante=True, cov="hessien", groupes=None,
            noms_x=None, noms_z=None, methode="bfgs"):
    s = np.asarray(s).astype(bool)
    y0 = np.where(s, np.asarray(y, dtype=float), 0.0)
    X, noms_x = donnees(X, constante, noms_x)
    Z, noms_z = donnees(Z, constante, noms_z)
    Kx, Kz = X.shape[1], Z.shape[1]

    def ll(theta):
        zg = Z @ theta[Kx:Kx + Kz]
        sigma, rho = np.exp(theta[-2]), np.tanh(theta[-1])
        e = (y0 - X @ theta[:Kx]) / sigma
        choisi = norm.logcdf((zg + rho * e) / np.sqrt(1 - rho ** 2)) + norm.logpdf(e) - np.log(sigma)
        return np.where(s, choisi, norm.logcdf(-zg))

    g0 = probit(s.astype(float), Z, constante=False).theta
    mills = norm.pdf(Z @ g0) / norm.cdf(Z @ g0)
    Xl = np.column_stack([X[s], mills[s]])
    coef = np.linalg.lstsq(Xl, y0[s], rcond=None)[0]
    sigma0 = np.sqrt(np.mean((y0[s] - Xl @ coef) ** 2))
    theta0 = np.concatenate([coef[:Kx], g0, [np.log(sigma0), 0.0]])

    noms = ["y:" + c for c in noms_x] + ["s:" + c for c in noms_z] + ["sigma", "rho"]
    res = estimer(
        ll, theta0, noms, cov, groupes, methode,
        transformation=lambda t: np.concatenate([t[:-2], [np.exp(t[-2]), np.tanh(t[-1])]]),
    )
    res.X, res.noms_x = X, noms_x
    return res
