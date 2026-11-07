from .var import ResultatVAR, choisir_ordre
from .structurel import irf_var, fevd_var, irf, fevd, irf_bootstrap, impact_cholesky, impact_long_terme
from .johansen import johansen, vecm, valeurs_critiques
from .kalman import filtre, lisseur, niveau_local, arma, cpp_disponible, init_diffuse, init_stationnaire


def var(Y, p, constante=True):
    return ResultatVAR(Y, p, constante)
