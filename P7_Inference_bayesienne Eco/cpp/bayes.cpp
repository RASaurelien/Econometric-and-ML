// Echantillonneurs bayesiens en C++ : Gibbs pour un panel lineaire a effets aleatoires
// et Metropolis-Hastings pour un logit. Memes algorithmes, meme generateur et memes ordres
// de calcul que les versions Rust et Python, pour obtenir exactement les memes tirages.
// Les matrices sont en ordre ligne.
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

typedef std::vector<double> V;

const double DEUX_PI = 6.283185307179586;

inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

// xoshiro256** initialise par SplitMix64. Les uniformes sont dans ]0, 1[.
struct Rng {
    uint64_t s[4];
    bool a_spare = false;
    double spare = 0.0;

    explicit Rng(uint64_t seed) {
        uint64_t z = seed;
        for (int i = 0; i < 4; i++) {
            z += 0x9E3779B97F4A7C15ULL;
            uint64_t x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            s[i] = x ^ (x >> 31);
        }
    }

    uint64_t suivant() {
        uint64_t r = rotl(s[1] * 5, 7) * 9;
        uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return r;
    }

    double uniforme() { return ((suivant() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

    // Box-Muller, on garde le deuxieme tirage pour l'appel suivant.
    double normale() {
        if (a_spare) {
            a_spare = false;
            return spare;
        }
        double u1 = uniforme(), u2 = uniforme();
        double r = std::sqrt(-2.0 * std::log(u1));
        double th = DEUX_PI * u2;
        spare = r * std::sin(th);
        a_spare = true;
        return r * std::cos(th);
    }

    // Marsaglia et Tsang pour Gamma(a, 1). Pour a < 1 on passe par Gamma(a + 1).
    double gamma(double a) {
        if (a < 1.0) {
            double g = gamma(a + 1.0);
            double u = uniforme();
            return g * std::pow(u, 1.0 / a);
        }
        double d = a - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
        for (;;) {
            double x = normale();
            double v = 1.0 + c * x;
            if (v <= 0.0) continue;
            v = v * v * v;
            double u = uniforme();
            if (std::log(u) < 0.5 * x * x + d - d * v + d * std::log(v)) return d * v;
        }
    }
};

// Cholesky : P = L L', L triangulaire inferieure (k x k).
void cholesky(const V& P, V& L, int k) {
    for (int i = 0; i < k; i++)
        for (int j = 0; j <= i; j++) {
            double s = P[i * k + j];
            for (int m = 0; m < j; m++) s -= L[i * k + m] * L[j * k + m];
            L[i * k + j] = (i == j) ? std::sqrt(s) : s / L[j * k + j];
        }
}

// Resout L' x = b (remontee).
void resoudre_lt(const V& L, const V& b, V& x, int k) {
    for (int i = k - 1; i >= 0; i--) {
        double s = b[i];
        for (int m = i + 1; m < k; m++) s -= L[m * k + i] * x[m];
        x[i] = s / L[i * k + i];
    }
}

// Resout L u = b (descente) puis L' x = u (remontee).
void resoudre_ll(const V& L, const V& b, V& x, int k) {
    V u(k);
    for (int i = 0; i < k; i++) {
        double s = b[i];
        for (int m = 0; m < i; m++) s -= L[i * k + m] * u[m];
        u[i] = s / L[i * k + i];
    }
    resoudre_lt(L, u, x, k);
}

// Log de la loi a posteriori du logit (a une constante pres), prior N(0, sd^2 I).
double log_posterieure(int n, int k, const double* y, const double* X, double sd, const V& b) {
    double ll = 0.0;
    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < k; j++) eta += X[i * k + j] * b[j];
        double sp = (eta > 0.0 ? eta : 0.0) + std::log1p(std::exp(-std::fabs(eta)));
        ll += y[i] * eta - sp;
    }
    double pr = 0.0;
    for (int j = 0; j < k; j++) pr += b[j] * b[j];
    return ll - 0.5 * pr / (sd * sd);
}

}  // namespace

// Premiers tirages du generateur, pour verifier que les trois implementations coincident :
// n uniformes, n normales, n Gamma(2.5) et n Gamma(0.4). out contient 4 n valeurs.
extern "C" void rng_flux(uint64_t seed, int n, double* out) {
    Rng r(seed);
    for (int i = 0; i < n; i++) out[i] = r.uniforme();
    for (int i = 0; i < n; i++) out[n + i] = r.normale();
    for (int i = 0; i < n; i++) out[2 * n + i] = r.gamma(2.5);
    for (int i = 0; i < n; i++) out[3 * n + i] = r.gamma(0.4);
}

// Gibbs pour y_i = x_i'b + a_{g(i)} + e_i, a_g ~ N(0, tau2), e ~ N(0, sigma2),
// b ~ N(0, var_beta I), sigma2 ~ IG(a_s, b_s), tau2 ~ IG(a_t, b_t).
// hyp = [var_beta, a_s, b_s, a_t, b_t], init = [b (k), a (ng), sigma2, tau2].
// out : n_garde lignes de k + 2 colonnes (b, sigma2, tau2). alpha_moyen : moyenne a posteriori des a_g.
extern "C" int gibbs_panel(int n, int k, int ng, const double* y, const double* X, const int* grp,
                           const double* hyp, const double* init, int n_iter, int burn, int thin,
                           uint64_t seed, double* out, double* alpha_moyen) {
    Rng rng(seed);
    double var_beta = hyp[0], a_s = hyp[1], b_s = hyp[2], a_t = hyp[3], b_t = hyp[4];
    V beta(init, init + k), alpha(init + k, init + k + ng);
    double sigma2 = init[k + ng], tau2 = init[k + ng + 1];

    V XtX(k * k, 0.0), Tg(ng, 0.0), S(ng), rhs(k), P(k * k), L(k * k), m(k), z(k), w(k);
    for (int i = 0; i < n; i++) {
        Tg[grp[i]] += 1.0;
        for (int a = 0; a < k; a++)
            for (int b = 0; b < k; b++) XtX[a * k + b] += X[i * k + a] * X[i * k + b];
    }
    for (int g = 0; g < ng; g++) alpha_moyen[g] = 0.0;
    int garde = 0;

    for (int it = 0; it < n_iter; it++) {
        for (int g = 0; g < ng; g++) S[g] = 0.0;
        for (int i = 0; i < n; i++) {
            double xb = 0.0;
            for (int j = 0; j < k; j++) xb += X[i * k + j] * beta[j];
            S[grp[i]] += y[i] - xb;
        }
        for (int g = 0; g < ng; g++) {
            double prec = Tg[g] / sigma2 + 1.0 / tau2;
            double moy = (S[g] / sigma2) / prec;
            alpha[g] = moy + rng.normale() / std::sqrt(prec);
        }

        for (int j = 0; j < k; j++) rhs[j] = 0.0;
        for (int i = 0; i < n; i++) {
            double r2 = y[i] - alpha[grp[i]];
            for (int j = 0; j < k; j++) rhs[j] += X[i * k + j] * r2;
        }
        for (int j = 0; j < k; j++) rhs[j] /= sigma2;
        for (int a = 0; a < k; a++)
            for (int b = 0; b < k; b++) P[a * k + b] = XtX[a * k + b] / sigma2 + (a == b ? 1.0 / var_beta : 0.0);
        cholesky(P, L, k);
        resoudre_ll(L, rhs, m, k);
        for (int j = 0; j < k; j++) z[j] = rng.normale();
        resoudre_lt(L, z, w, k);
        for (int j = 0; j < k; j++) beta[j] = m[j] + w[j];

        double ss = 0.0;
        for (int i = 0; i < n; i++) {
            double xb = 0.0;
            for (int j = 0; j < k; j++) xb += X[i * k + j] * beta[j];
            double r = y[i] - xb - alpha[grp[i]];
            ss += r * r;
        }
        sigma2 = (b_s + 0.5 * ss) / rng.gamma(a_s + 0.5 * n);

        double sa = 0.0;
        for (int g = 0; g < ng; g++) sa += alpha[g] * alpha[g];
        tau2 = (b_t + 0.5 * sa) / rng.gamma(a_t + 0.5 * ng);

        if (it >= burn && (it - burn) % thin == 0) {
            for (int j = 0; j < k; j++) out[garde * (k + 2) + j] = beta[j];
            out[garde * (k + 2) + k] = sigma2;
            out[garde * (k + 2) + k + 1] = tau2;
            for (int g = 0; g < ng; g++) alpha_moyen[g] += alpha[g];
            garde++;
        }
    }
    for (int g = 0; g < ng; g++) alpha_moyen[g] /= garde;
    return 0;
}

// Metropolis-Hastings a marche aleatoire pour un logit avec prior N(0, sd^2 I).
// La proposition est b + L z avec z ~ N(0, I), L triangulaire inferieure fournie (k x k).
// out : n_garde lignes de k colonnes. accepte : nombre total de propositions acceptees.
extern "C" int mh_logit(int n, int k, const double* y, const double* X, double sd, const double* L,
                        const double* x0, int n_iter, int burn, int thin, uint64_t seed, double* out,
                        int* accepte) {
    Rng rng(seed);
    V b(x0, x0 + k), prop(k), z(k);
    double lp = log_posterieure(n, k, y, X, sd, b);
    int garde = 0, acc = 0;
    for (int it = 0; it < n_iter; it++) {
        for (int j = 0; j < k; j++) z[j] = rng.normale();
        for (int i = 0; i < k; i++) {
            double s = 0.0;
            for (int j = 0; j <= i; j++) s += L[i * k + j] * z[j];
            prop[i] = b[i] + s;
        }
        double lp2 = log_posterieure(n, k, y, X, sd, prop);
        double u = rng.uniforme();
        if (std::log(u) < lp2 - lp) {
            b = prop;
            lp = lp2;
            acc++;
        }
        if (it >= burn && (it - burn) % thin == 0) {
            for (int j = 0; j < k; j++) out[garde * k + j] = b[j];
            garde++;
        }
    }
    *accepte = acc;
    return 0;
}
