import numpy as np


# Autocovariance par FFT, normalisee par n (comme Stan).
def autocov(x):
    n = len(x)
    f = np.fft.rfft(x - x.mean(), 2 * n)
    return np.fft.irfft(f * np.conj(f))[:n] / n


# R-hat decoupe (chaque chaine est coupee en deux). chaines a la forme (m, n).
# Une valeur proche de 1 (moins de 1.01) indique que les chaines ont converge vers la meme loi.
def rhat(chaines):
    m, n = chaines.shape
    h = n // 2
    c = np.vstack([chaines[:, :h], chaines[:, h:2 * h]])
    W = c.var(axis=1, ddof=1).mean()
    B = h * c.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((h - 1) / h * W + B / h) / W))


# Taille d'echantillon effective, avec la regle de Geyer sur les sommes de paires
# d'autocorrelations (meme methode que Stan, sans normalisation par les rangs).
def ess(chaines):
    m, n = chaines.shape
    W = chaines.var(axis=1, ddof=1).mean()
    B = n * chaines.mean(axis=1).var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * W + B / n
    ac = np.mean([autocov(c) for c in chaines], axis=0)
    rho = 1 - (W - ac) / var_plus
    rho[0] = 1.0
    paires, precedent, t = [], np.inf, 0
    while 2 * t + 1 < n:
        P = rho[2 * t] + rho[2 * t + 1]
        if P < 0:
            break
        precedent = min(P, precedent)
        paires.append(precedent)
        t += 1
    tau = max(-1 + 2 * sum(paires), 1 / np.log10(m * n))
    return float(m * n / tau)


# Resume a posteriori d'un tableau de tirages (chaines, n, parametres).
class Resume:
    def __init__(self, tirages, noms, niveau=0.95):
        a = (1 - niveau) / 2
        self.noms = list(noms)
        self.lignes = []
        for j in range(tirages.shape[2]):
            x = tirages[:, :, j]
            bas, haut = np.quantile(x, [a, 1 - a])
            self.lignes.append([x.mean(), x.std(ddof=1), bas, haut, ess(x), rhat(x)])
        self.niveau = niveau

    def __str__(self):
        pct = int(round(100 * self.niveau))
        tete = f"{'':>10}{'moyenne':>10}{'ecart-type':>12}{'bas ' + str(pct) + '%':>10}{'haut ' + str(pct) + '%':>10}{'ESS':>9}{'R-hat':>8}"
        sortie = [tete]
        for nom, l in zip(self.noms, self.lignes):
            sortie.append(f"{nom:>10}{l[0]:>10.4f}{l[1]:>12.4f}{l[2]:>10.4f}{l[3]:>10.4f}{l[4]:>9.0f}{l[5]:>8.3f}")
        return "\n".join(sortie)
