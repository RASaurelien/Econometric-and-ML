import math

MASQUE = (1 << 64) - 1
DEUX_PI = 6.283185307179586


def rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & MASQUE


# xoshiro256** initialise par SplitMix64, identique aux versions C++ et Rust.
# Les uniformes sont dans ]0, 1[. Les entiers de Python sont non bornes, d'ou les masques.
class Rng:
    def __init__(self, graine):
        z = graine & MASQUE
        self.s = []
        for _ in range(4):
            z = (z + 0x9E3779B97F4A7C15) & MASQUE
            x = z
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASQUE
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASQUE
            self.s.append(x ^ (x >> 31))
        self.a_spare = False
        self.spare = 0.0

    def suivant(self):
        s = self.s
        r = (rotl((s[1] * 5) & MASQUE, 7) * 9) & MASQUE
        t = (s[1] << 17) & MASQUE
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        return r

    def uniforme(self):
        return ((self.suivant() >> 11) + 0.5) * (1.0 / 9007199254740992.0)

    # Box-Muller, on garde le deuxieme tirage pour l'appel suivant.
    def normale(self):
        if self.a_spare:
            self.a_spare = False
            return self.spare
        u1, u2 = self.uniforme(), self.uniforme()
        r = math.sqrt(-2.0 * math.log(u1))
        th = DEUX_PI * u2
        self.spare = r * math.sin(th)
        self.a_spare = True
        return r * math.cos(th)

    # Marsaglia et Tsang pour Gamma(a, 1). Pour a < 1 on passe par Gamma(a + 1).
    def gamma(self, a):
        if a < 1.0:
            g = self.gamma(a + 1.0)
            u = self.uniforme()
            return g * math.pow(u, 1.0 / a)
        d = a - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.normale()
            v = 1.0 + c * x
            if v <= 0.0:
                continue
            v = v * v * v
            u = self.uniforme()
            if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
                return d * v
