import numpy as np


# Regresseur centre reduit. "asymetrique" : lognormale, ce qui donne quelques
# points tres influents (fort levier), le cas ou les variantes de White different.
def tirer_x(rng, forme, taille):
    z = rng.normal(size=taille)
    if forme == "normal":
        return z
    moyenne, variance = np.exp(0.32), (np.exp(0.64) - 1) * np.exp(0.64)
    return (np.exp(0.8 * z) - moyenne) / np.sqrt(variance)


# Erreurs de moyenne 0 et de variance 1.
def tirer_erreurs(rng, forme, taille):
    if forme == "normale":
        return rng.normal(size=taille)
    if forme == "student":
        return rng.standard_t(5, size=taille) / np.sqrt(5 / 3)
    return (rng.chisquare(1, size=taille) - 1) / np.sqrt(2)


# R echantillons de taille n du modele y = b0 + b1 x + sigma(x) u,
# avec sigma(x) = exp(hetero * x). hetero = 0 donne des erreurs homoscedastiques.
def generer(R, n, rng, beta=(1.0, 0.5), hetero=0.0, regresseurs="normal", erreurs="normale"):
    x = tirer_x(rng, regresseurs, (R, n))
    u = tirer_erreurs(rng, erreurs, (R, n))
    y = beta[0] + beta[1] * x + np.exp(hetero * x) * u
    return np.stack([np.ones((R, n)), x], axis=2), y


# Modele avec un regresseur endogene et m instruments, sans constante.
# y = beta x + e, x = z'pi + v, e = rho v + sqrt(1 - rho^2) u.
# La force est donnee par le parametre de concentration mu2 = n pi'pi.
def generer_iv(R, n, m, mu2, rng, rho=0.8, beta=1.0):
    z = rng.normal(size=(R, n, m))
    v, u = rng.normal(size=(R, n)), rng.normal(size=(R, n))
    x = z.sum(axis=2) * np.sqrt(mu2 / (n * m)) + v
    e = rho * v + np.sqrt(1 - rho ** 2) * u
    return beta * x + e, x, z


# Meme modele avec une force fixe, endogene ou non selon l'argument.
def generer_endogene(R, n, rng, endogene=True, m=3, beta=1.0):
    z = rng.normal(size=(R, n, m))
    v, u = rng.normal(size=(R, n)), rng.normal(size=(R, n))
    x = 0.4 * z.sum(axis=2) + v
    e = 0.6 * v + 0.8 * u if endogene else u
    return beta * x + e, x, z
