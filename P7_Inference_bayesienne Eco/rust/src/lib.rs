// Echantillonneurs bayesiens en Rust : Gibbs pour un panel lineaire a effets aleatoires
// et Metropolis-Hastings pour un logit. Memes algorithmes, meme generateur et memes ordres
// de calcul que les versions C++ et Python, pour obtenir exactement les memes tirages.
// Les matrices sont en ordre ligne.
use std::slice;

const DEUX_PI: f64 = 6.283185307179586;

// xoshiro256** initialise par SplitMix64. Les uniformes sont dans ]0, 1[.
struct Rng {
    s: [u64; 4],
    a_spare: bool,
    spare: f64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut s = [0u64; 4];
        for i in 0..4 {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            s[i] = x ^ (x >> 31);
        }
        Rng { s, a_spare: false, spare: 0.0 }
    }

    fn suivant(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    fn uniforme(&mut self) -> f64 {
        ((self.suivant() >> 11) as f64 + 0.5) * (1.0 / 9007199254740992.0)
    }

    // Box-Muller, on garde le deuxieme tirage pour l'appel suivant.
    fn normale(&mut self) -> f64 {
        if self.a_spare {
            self.a_spare = false;
            return self.spare;
        }
        let u1 = self.uniforme();
        let u2 = self.uniforme();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = DEUX_PI * u2;
        self.spare = r * th.sin();
        self.a_spare = true;
        r * th.cos()
    }

    // Marsaglia et Tsang pour Gamma(a, 1). Pour a < 1 on passe par Gamma(a + 1).
    fn gamma(&mut self, a: f64) -> f64 {
        if a < 1.0 {
            let g = self.gamma(a + 1.0);
            let u = self.uniforme();
            return g * u.powf(1.0 / a);
        }
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normale();
            let mut v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            v = v * v * v;
            let u = self.uniforme();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

// Cholesky : P = L L', L triangulaire inferieure (k x k).
fn cholesky(p: &[f64], l: &mut [f64], k: usize) {
    for i in 0..k {
        for j in 0..=i {
            let mut s = p[i * k + j];
            for m in 0..j {
                s -= l[i * k + m] * l[j * k + m];
            }
            l[i * k + j] = if i == j { s.sqrt() } else { s / l[j * k + j] };
        }
    }
}

// Resout L' x = b (remontee).
fn resoudre_lt(l: &[f64], b: &[f64], x: &mut [f64], k: usize) {
    for i in (0..k).rev() {
        let mut s = b[i];
        for m in (i + 1)..k {
            s -= l[m * k + i] * x[m];
        }
        x[i] = s / l[i * k + i];
    }
}

// Resout L u = b (descente) puis L' x = u (remontee).
fn resoudre_ll(l: &[f64], b: &[f64], x: &mut [f64], k: usize) {
    let mut u = vec![0.0; k];
    for i in 0..k {
        let mut s = b[i];
        for m in 0..i {
            s -= l[i * k + m] * u[m];
        }
        u[i] = s / l[i * k + i];
    }
    resoudre_lt(l, &u, x, k);
}

// Log de la loi a posteriori du logit (a une constante pres), prior N(0, sd^2 I).
fn log_posterieure(n: usize, k: usize, y: &[f64], x: &[f64], sd: f64, b: &[f64]) -> f64 {
    let mut ll = 0.0;
    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..k {
            eta += x[i * k + j] * b[j];
        }
        let sp = (if eta > 0.0 { eta } else { 0.0 }) + (-eta.abs()).exp().ln_1p();
        ll += y[i] * eta - sp;
    }
    let mut pr = 0.0;
    for j in 0..k {
        pr += b[j] * b[j];
    }
    ll - 0.5 * pr / (sd * sd)
}

// Premiers tirages du generateur, pour verifier que les trois implementations coincident :
// n uniformes, n normales, n Gamma(2.5) et n Gamma(0.4). out contient 4 n valeurs.
#[no_mangle]
pub unsafe extern "C" fn rng_flux(seed: u64, n: i32, out: *mut f64) {
    let n = n as usize;
    let out = slice::from_raw_parts_mut(out, 4 * n);
    let mut r = Rng::new(seed);
    for i in 0..n {
        out[i] = r.uniforme();
    }
    for i in 0..n {
        out[n + i] = r.normale();
    }
    for i in 0..n {
        out[2 * n + i] = r.gamma(2.5);
    }
    for i in 0..n {
        out[3 * n + i] = r.gamma(0.4);
    }
}

// Gibbs pour y_i = x_i'b + a_{g(i)} + e_i, a_g ~ N(0, tau2), e ~ N(0, sigma2),
// b ~ N(0, var_beta I), sigma2 ~ IG(a_s, b_s), tau2 ~ IG(a_t, b_t).
// hyp = [var_beta, a_s, b_s, a_t, b_t], init = [b (k), a (ng), sigma2, tau2].
// out : n_garde lignes de k + 2 colonnes (b, sigma2, tau2). alpha_moyen : moyenne a posteriori des a_g.
#[no_mangle]
pub unsafe extern "C" fn gibbs_panel(
    n: i32, k: i32, ng: i32, y: *const f64, x: *const f64, grp: *const i32, hyp: *const f64,
    init: *const f64, n_iter: i32, burn: i32, thin: i32, seed: u64, out: *mut f64,
    alpha_moyen: *mut f64,
) -> i32 {
    let (n, k, ng) = (n as usize, k as usize, ng as usize);
    let (n_iter, burn, thin) = (n_iter as usize, burn as usize, thin as usize);
    let y = slice::from_raw_parts(y, n);
    let x = slice::from_raw_parts(x, n * k);
    let grp = slice::from_raw_parts(grp, n);
    let hyp = slice::from_raw_parts(hyp, 5);
    let init = slice::from_raw_parts(init, k + ng + 2);
    let n_garde_max = if n_iter > burn { (n_iter - burn + thin - 1) / thin } else { 0 };
    let out = slice::from_raw_parts_mut(out, n_garde_max * (k + 2));
    let alpha_moyen = slice::from_raw_parts_mut(alpha_moyen, ng);

    let mut rng = Rng::new(seed);
    let (var_beta, a_s, b_s, a_t, b_t) = (hyp[0], hyp[1], hyp[2], hyp[3], hyp[4]);
    let mut beta: Vec<f64> = init[..k].to_vec();
    let mut alpha: Vec<f64> = init[k..k + ng].to_vec();
    let mut sigma2 = init[k + ng];
    let mut tau2 = init[k + ng + 1];

    let mut xtx = vec![0.0; k * k];
    let mut tg = vec![0.0; ng];
    let mut s = vec![0.0; ng];
    let mut rhs = vec![0.0; k];
    let mut p = vec![0.0; k * k];
    let mut l = vec![0.0; k * k];
    let mut m = vec![0.0; k];
    let mut z = vec![0.0; k];
    let mut w = vec![0.0; k];
    for i in 0..n {
        tg[grp[i] as usize] += 1.0;
        for a in 0..k {
            for b in 0..k {
                xtx[a * k + b] += x[i * k + a] * x[i * k + b];
            }
        }
    }
    for g in 0..ng {
        alpha_moyen[g] = 0.0;
    }
    let mut garde = 0usize;

    for it in 0..n_iter {
        for g in 0..ng {
            s[g] = 0.0;
        }
        for i in 0..n {
            let mut xb = 0.0;
            for j in 0..k {
                xb += x[i * k + j] * beta[j];
            }
            s[grp[i] as usize] += y[i] - xb;
        }
        for g in 0..ng {
            let prec = tg[g] / sigma2 + 1.0 / tau2;
            let moy = (s[g] / sigma2) / prec;
            alpha[g] = moy + rng.normale() / prec.sqrt();
        }

        for j in 0..k {
            rhs[j] = 0.0;
        }
        for i in 0..n {
            let r2 = y[i] - alpha[grp[i] as usize];
            for j in 0..k {
                rhs[j] += x[i * k + j] * r2;
            }
        }
        for j in 0..k {
            rhs[j] /= sigma2;
        }
        for a in 0..k {
            for b in 0..k {
                p[a * k + b] = xtx[a * k + b] / sigma2 + if a == b { 1.0 / var_beta } else { 0.0 };
            }
        }
        cholesky(&p, &mut l, k);
        resoudre_ll(&l, &rhs, &mut m, k);
        for j in 0..k {
            z[j] = rng.normale();
        }
        resoudre_lt(&l, &z, &mut w, k);
        for j in 0..k {
            beta[j] = m[j] + w[j];
        }

        let mut ss = 0.0;
        for i in 0..n {
            let mut xb = 0.0;
            for j in 0..k {
                xb += x[i * k + j] * beta[j];
            }
            let r = y[i] - xb - alpha[grp[i] as usize];
            ss += r * r;
        }
        sigma2 = (b_s + 0.5 * ss) / rng.gamma(a_s + 0.5 * n as f64);

        let mut sa = 0.0;
        for g in 0..ng {
            sa += alpha[g] * alpha[g];
        }
        tau2 = (b_t + 0.5 * sa) / rng.gamma(a_t + 0.5 * ng as f64);

        if it >= burn && (it - burn) % thin == 0 {
            for j in 0..k {
                out[garde * (k + 2) + j] = beta[j];
            }
            out[garde * (k + 2) + k] = sigma2;
            out[garde * (k + 2) + k + 1] = tau2;
            for g in 0..ng {
                alpha_moyen[g] += alpha[g];
            }
            garde += 1;
        }
    }
    for g in 0..ng {
        alpha_moyen[g] /= garde as f64;
    }
    0
}

// Metropolis-Hastings a marche aleatoire pour un logit avec prior N(0, sd^2 I).
// La proposition est b + L z avec z ~ N(0, I), L triangulaire inferieure fournie (k x k).
// out : n_garde lignes de k colonnes. accepte : nombre total de propositions acceptees.
#[no_mangle]
pub unsafe extern "C" fn mh_logit(
    n: i32, k: i32, y: *const f64, x: *const f64, sd: f64, l: *const f64, x0: *const f64,
    n_iter: i32, burn: i32, thin: i32, seed: u64, out: *mut f64, accepte: *mut i32,
) -> i32 {
    let (n, k) = (n as usize, k as usize);
    let (n_iter, burn, thin) = (n_iter as usize, burn as usize, thin as usize);
    let y = slice::from_raw_parts(y, n);
    let x = slice::from_raw_parts(x, n * k);
    let l = slice::from_raw_parts(l, k * k);
    let n_garde_max = if n_iter > burn { (n_iter - burn + thin - 1) / thin } else { 0 };
    let out = slice::from_raw_parts_mut(out, n_garde_max * k);

    let mut rng = Rng::new(seed);
    let mut b: Vec<f64> = slice::from_raw_parts(x0, k).to_vec();
    let mut prop = vec![0.0; k];
    let mut z = vec![0.0; k];
    let mut lp = log_posterieure(n, k, y, x, sd, &b);
    let mut garde = 0usize;
    let mut acc = 0;
    for it in 0..n_iter {
        for j in 0..k {
            z[j] = rng.normale();
        }
        for i in 0..k {
            let mut s = 0.0;
            for j in 0..=i {
                s += l[i * k + j] * z[j];
            }
            prop[i] = b[i] + s;
        }
        let lp2 = log_posterieure(n, k, y, x, sd, &prop);
        let u = rng.uniforme();
        if u.ln() < lp2 - lp {
            b.copy_from_slice(&prop);
            lp = lp2;
            acc += 1;
        }
        if it >= burn && (it - burn) % thin == 0 {
            for j in 0..k {
                out[garde * k + j] = b[j];
            }
            garde += 1;
        }
    }
    *accepte = acc;
    0
}
