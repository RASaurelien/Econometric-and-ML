import numpy as np
from scipy.stats import norm

# Noyaux usuels, tous d'integrale 1.
NOYAUX = {
    "gaussien": norm.pdf,
    "epanechnikov": lambda u: 0.75 * (1 - u ** 2) * (np.abs(u) <= 1),
    "triangulaire": lambda u: (1 - np.abs(u)) * (np.abs(u) <= 1),
    "uniforme": lambda u: 0.5 * (np.abs(u) <= 1),
}
