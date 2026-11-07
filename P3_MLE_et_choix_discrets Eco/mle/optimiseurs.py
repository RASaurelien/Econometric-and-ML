import numpy as np


# Backtracking : on divise le pas par 2 jusqu'a obtenir une baisse suffisante.
# La condition est ecrite pour qu'un nan reduise aussi le pas.
def recherche_pas(f, x, d, g, f0):
    t = 1.0
    while not f(x + t * d) <= f0 + 1e-4 * t * (g @ d) and t > 1e-12:
        t /= 2
    return t


# Newton-Raphson pour minimiser f. On s'arrete quand le decrement de Newton
# g' H^-1 g est tres petit. Si H n'est pas definie positive, on prend -g.
def newton(f, grad, hess, x0, tol=1e-10, maxit=100):
    x = np.asarray(x0, dtype=float)
    for it in range(maxit):
        g = grad(x)
        try:
            d = -np.linalg.solve(hess(x), g)
        except np.linalg.LinAlgError:
            d = -g
        if g @ d >= 0:
            d = -g
        if -(g @ d) < tol:
            return x, it, True
        x = x + recherche_pas(f, x, d, g, f(x)) * d
    return x, maxit, False


# BFGS pour minimiser f. On met a jour directement l'inverse du hessien.
def bfgs(f, grad, x0, tol=1e-8, maxit=500):
    x = np.asarray(x0, dtype=float)
    n = len(x)
    Hinv = np.eye(n)
    g = grad(x)
    for it in range(maxit):
        if np.max(np.abs(g)) <= tol * max(1.0, abs(f(x))):
            return x, it, True
        d = -Hinv @ g
        if g @ d >= 0:
            Hinv, d = np.eye(n), -g
        x_new = x + recherche_pas(f, x, d, g, f(x)) * d
        g_new = grad(x_new)
        s, y = x_new - x, g_new - g
        if s @ y > 1e-12:
            rho = 1 / (s @ y)
            I = np.eye(n)
            Hinv = (I - rho * np.outer(s, y)) @ Hinv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, g = x_new, g_new
    return x, maxit, False
