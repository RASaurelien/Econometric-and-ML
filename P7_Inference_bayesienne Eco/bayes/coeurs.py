import ctypes
import os

import numpy as np

from . import reference

DOSSIER = os.path.dirname(os.path.abspath(__file__))


# Charge une bibliotheque compilee (make), ou renvoie None si elle est absente.
def charger(nom):
    try:
        lib = ctypes.CDLL(os.path.join(DOSSIER, nom))
    except OSError:
        return None
    p, i, d, u = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_uint64
    lib.rng_flux.argtypes = [u, i, p]
    lib.rng_flux.restype = None
    lib.gibbs_panel.argtypes = [i, i, i, p, p, p, p, p, i, i, i, u, p, p]
    lib.gibbs_panel.restype = i
    lib.mh_logit.argtypes = [i, i, p, p, d, p, p, i, i, i, u, p, p]
    lib.mh_logit.restype = i
    return lib


LIBS = {"cpp": charger("libbayes_cpp.so"), "rust": charger("libbayes_rust.so")}


# Cœurs utilisables, du plus rapide au plus lent en general.
def disponibles():
    return [nom for nom in ("rust", "cpp") if LIBS[nom] is not None] + ["python"]


def choisir(backend):
    if backend == "auto":
        return disponibles()[0]
    if backend != "python" and LIBS.get(backend) is None:
        raise RuntimeError(f"cœur {backend} absent, lancer make")
    return backend


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def tableau(a, dtype=float):
    return np.ascontiguousarray(a, dtype=dtype)


def n_garde(n_iter, burn, thin):
    if n_iter <= burn:
        raise ValueError("n_iter doit etre superieur a burn")
    return (n_iter - burn + thin - 1) // thin


def flux(backend, graine, n):
    if backend == "python":
        return reference.rng_flux(graine, n)
    out = np.empty(4 * n)
    LIBS[backend].rng_flux(graine, n, ptr(out))
    return out


# Gibbs du panel a effets aleatoires. Renvoie les tirages (n_garde, k + 2) et la moyenne des effets.
def gibbs(backend, y, X, grp, ng, init, hyp, n_iter, burn, thin, graine):
    y, X, init, hyp = tableau(y), tableau(X), tableau(init), tableau(hyp)
    grp = tableau(grp, np.int32)
    ng_garde = n_garde(n_iter, burn, thin)
    if backend == "python":
        return reference.gibbs_panel(y, X, grp, ng, hyp, init, n_iter, burn, thin, graine)
    n, k = X.shape
    out, alpha = np.empty((ng_garde, k + 2)), np.empty(ng)
    code = LIBS[backend].gibbs_panel(n, k, ng, ptr(y), ptr(X), ptr(grp), ptr(hyp), ptr(init),
                                     n_iter, burn, thin, graine, ptr(out), ptr(alpha))
    if code:
        raise RuntimeError("echec du cœur " + backend)
    return out, alpha


# Metropolis-Hastings du logit. Renvoie les tirages (n_garde, k) et le nombre d'acceptations.
def mh(backend, y, X, sd, L, x0, n_iter, burn, thin, graine):
    y, X, L, x0 = tableau(y), tableau(X), tableau(L), tableau(x0)
    ng_garde = n_garde(n_iter, burn, thin)
    if backend == "python":
        return reference.mh_logit(y, X, sd, L, x0, n_iter, burn, thin, graine)
    n, k = X.shape
    out, acc = np.empty((ng_garde, k)), ctypes.c_int(0)
    code = LIBS[backend].mh_logit(n, k, ptr(y), ptr(X), sd, ptr(L), ptr(x0), n_iter, burn, thin, graine,
                                  ptr(out), ctypes.byref(acc))
    if code:
        raise RuntimeError("echec du cœur " + backend)
    return out, acc.value
