// Filtre de Kalman pour un modele lineaire gaussien invariant dans le temps :
//   y_t     = Z a_t + d + eps_t,         eps ~ N(0, H)
//   a_{t+1} = T a_t + c + R eta_t,       eta ~ N(0, Q)
// Le produit R Q R' (RQR) est calcule cote Python. Les matrices sont en ordre ligne.
// Les valeurs manquantes (NaN) de y sont ignorees, y compris partiellement.
#include <cmath>
#include <vector>

namespace {

typedef std::vector<double> V;

// C (n x p) = A (n x m) * B (m x p)
void produit(const double* A, const double* B, double* C, int n, int m, int p) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            double s = 0;
            for (int k = 0; k < m; k++) s += A[i * m + k] * B[k * p + j];
            C[i * p + j] = s;
        }
}

// C (n x p) = A (n x m) * B' avec B (p x m)
void produit_bt(const double* A, const double* B, double* C, int n, int m, int p) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            double s = 0;
            for (int k = 0; k < m; k++) s += A[i * m + k] * B[j * m + k];
            C[i * p + j] = s;
        }
}

// Inverse d'une matrice symetrique definie positive par Cholesky.
// Renvoie le log du determinant, ou NAN si la matrice n'est pas definie positive.
double inverse_spd(const V& F, V& Finv, int q) {
    V L(q * q, 0.0), x(q);
    double logdet = 0;
    for (int i = 0; i < q; i++)
        for (int j = 0; j <= i; j++) {
            double s = F[i * q + j];
            for (int k = 0; k < j; k++) s -= L[i * q + k] * L[j * q + k];
            if (i == j) {
                if (s <= 0) return NAN;
                L[i * q + i] = std::sqrt(s);
                logdet += 2 * std::log(L[i * q + i]);
            } else {
                L[i * q + j] = s / L[j * q + j];
            }
        }
    for (int col = 0; col < q; col++) {
        for (int i = 0; i < q; i++) {
            double s = (i == col) ? 1.0 : 0.0;
            for (int k = 0; k < i; k++) s -= L[i * q + k] * x[k];
            x[i] = s / L[i * q + i];
        }
        for (int i = q - 1; i >= 0; i--) {
            double s = x[i];
            for (int k = i + 1; k < q; k++) s -= L[k * q + i] * x[k];
            x[i] = s / L[i * q + i];
        }
        for (int i = 0; i < q; i++) Finv[i * q + col] = x[i];
    }
    return logdet;
}

void symetriser(V& P, int m) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < i; j++) P[i * m + j] = P[j * m + i] = 0.5 * (P[i * m + j] + P[j * m + i]);
}

}  // namespace

// n observations, q variables observees, m etats. a0 et P0 : loi de l'etat avant y_0.
// Les sorties a_pred, P_pred (n+1 lignes) et a_filt, P_filt (n lignes) peuvent etre nulles
// quand on ne veut que la vraisemblance. Renvoie 0 si tout va bien, 1 si F n'est pas inversible.
extern "C" int kalman_filtre(int n, int q, int m, const double* y, const double* Z, const double* d,
                             const double* H, const double* Tm, const double* c, const double* RQR,
                             const double* a0, const double* P0, double* a_pred, double* P_pred,
                             double* a_filt, double* P_filt, double* loglik) {
    V a(a0, a0 + m), P(P0, P0 + m * m), af(m), Pf(m * m), tmp(m * m), P2(m * m);
    V Zs(q * m), Hs(q * q), v(q), Fm(q * q), Finv(q * q), PZt(m * q), K(m * q);
    std::vector<int> obs(q);
    const double log2pi = std::log(2 * M_PI);
    double ll = 0;

    for (int t = 0; t < n; t++) {
        if (a_pred) {
            for (int i = 0; i < m; i++) a_pred[t * m + i] = a[i];
            for (int i = 0; i < m * m; i++) P_pred[t * m * m + i] = P[i];
        }
        int qo = 0;
        for (int i = 0; i < q; i++)
            if (!std::isnan(y[t * q + i])) obs[qo++] = i;

        if (qo == 0) {
            af = a;
            Pf = P;
        } else {
            for (int r = 0; r < qo; r++) {
                for (int k = 0; k < m; k++) Zs[r * m + k] = Z[obs[r] * m + k];
                for (int s = 0; s < qo; s++) Hs[r * qo + s] = H[obs[r] * q + obs[s]];
                double zr = 0;
                for (int k = 0; k < m; k++) zr += Zs[r * m + k] * a[k];
                v[r] = y[t * q + obs[r]] - d[obs[r]] - zr;
            }
            produit_bt(P.data(), Zs.data(), PZt.data(), m, m, qo);
            produit(Zs.data(), PZt.data(), Fm.data(), qo, m, qo);
            for (int i = 0; i < qo * qo; i++) Fm[i] += Hs[i];
            double logdet = inverse_spd(Fm, Finv, qo);
            if (std::isnan(logdet)) return 1;
            produit(PZt.data(), Finv.data(), K.data(), m, qo, qo);
            double quad = 0;
            for (int r = 0; r < qo; r++)
                for (int s = 0; s < qo; s++) quad += v[r] * Finv[r * qo + s] * v[s];
            for (int i = 0; i < m; i++) {
                double s = a[i];
                for (int r = 0; r < qo; r++) s += K[i * qo + r] * v[r];
                af[i] = s;
            }
            produit_bt(K.data(), PZt.data(), tmp.data(), m, qo, m);
            for (int i = 0; i < m * m; i++) Pf[i] = P[i] - tmp[i];
            symetriser(Pf, m);
            ll += -0.5 * (qo * log2pi + logdet + quad);
        }
        if (a_filt) {
            for (int i = 0; i < m; i++) a_filt[t * m + i] = af[i];
            for (int i = 0; i < m * m; i++) P_filt[t * m * m + i] = Pf[i];
        }
        for (int i = 0; i < m; i++) {
            double s = c[i];
            for (int k = 0; k < m; k++) s += Tm[i * m + k] * af[k];
            a[i] = s;
        }
        produit(Tm, Pf.data(), tmp.data(), m, m, m);
        produit_bt(tmp.data(), Tm, P2.data(), m, m, m);
        for (int i = 0; i < m * m; i++) P[i] = P2[i] + RQR[i];
        symetriser(P, m);
    }
    if (a_pred) {
        for (int i = 0; i < m; i++) a_pred[n * m + i] = a[i];
        for (int i = 0; i < m * m; i++) P_pred[n * m * m + i] = P[i];
    }
    *loglik = ll;
    return 0;
}
