# Discontinuite de regression nette : polynome local, correction du biais et variance robuste
# (Calonico, Cattaneo et Titiunik, 2014). Noyau triangulaire, variance par plus proches voisins.

# Residus par plus proches voisins (les ex aequo sont regroupes)
rd_nn_res <- function(x, y, J = 3) {
  n <- length(x)
  if (n <= 1) return(rep(0, n))
  o <- order(x)
  xs <- x[o]
  ys <- y[o]
  r <- rle(xs)
  dups <- rep(r$lengths, r$lengths)
  dupsid <- sequence(r$lengths)
  besoin <- min(J, n - 1)
  res <- numeric(n)
  for (pos in seq_len(n)) {
    rpos <- dups[pos] - dupsid[pos]
    lpos <- dupsid[pos] - 1
    while (lpos + rpos < besoin) {
      if (pos - lpos - 1 <= 0) {
        rpos <- rpos + dups[pos + rpos + 1]
      } else if (pos + rpos + 1 > n) {
        lpos <- lpos + dups[pos - lpos - 1]
      } else {
        dl <- xs[pos] - xs[pos - lpos - 1]
        dr <- xs[pos + rpos + 1] - xs[pos]
        if (dl > dr) {
          rpos <- rpos + dups[pos + rpos + 1]
        } else if (dl < dr) {
          lpos <- lpos + dups[pos - lpos - 1]
        } else {
          rpos <- rpos + dups[pos + rpos + 1]
          lpos <- lpos + dups[pos - lpos - 1]
        }
      }
    }
    idx <- (pos - lpos):(pos + rpos)
    Ji <- length(idx) - 1
    res[pos] <- sqrt(Ji / (Ji + 1)) * (ys[pos] - (sum(ys[idx]) - ys[pos]) / Ji)
  }
  out <- numeric(n)
  out[o] <- res
  out
}

# Poids lineaires du coefficient j d'un polynome local d'ordre p (noyau triangulaire, fenetre h)
rd_poids <- function(x, c, h, p, j) {
  u <- (x - c) / h
  w <- pmax(0, 1 - abs(u))
  idx <- which(w > 0)
  if (length(idx) <= p) stop("pas assez d'observations dans la fenetre")
  R <- outer(u[idx], 0:p, "^")
  Rw <- R * w[idx]
  Ginv <- solve(crossprod(R, Rw))
  list(idx = idx, a = drop(Rw %*% Ginv[, j + 1]) / h^j)
}

# Estimations d'un seul cote du seuil
rd_cote <- function(x, y, c, h, b, p, q, nn) {
  dans <- which(abs(x - c) <= max(h, b))
  xs <- x[dans]
  ys <- y[dans]
  sig2 <- rd_nn_res(xs, ys, nn)^2
  ah <- rd_poids(xs, c, h, p, 0)
  ab <- rd_poids(xs, c, b, q, p + 1)
  psi <- sum(ah$a * (xs[ah$idx] - c)^(p + 1))
  tau <- sum(ah$a * ys[ah$idx])
  biais <- psi * sum(ab$a * ys[ab$idx])
  cl <- numeric(length(xs))
  cl[ah$idx] <- ah$a
  rb <- cl
  rb[ab$idx] <- rb[ab$idx] - psi * ab$a
  list(tau = tau, tau_bc = tau - biais, v_cl = sum(cl^2 * sig2), v_rb = sum(rb^2 * sig2),
       n_h = sum(abs(x - c) <= h))
}

rd_robust <- function(y, x, c = 0, h = NULL, b = NULL, p = 1, q = p + 1, nn = 3, niveau = 0.95) {
  ok <- !is.na(y) & !is.na(x)
  y <- y[ok]
  x <- x[ok]
  if (is.null(h)) {
    bw <- rd_bandwidth(y, x, c, p, q, nn)
    h <- bw[["h"]]
    if (is.null(b)) b <- bw[["b"]]
  }
  if (is.null(b)) b <- h
  g <- x < c
  gau <- rd_cote(x[g], y[g], c, h, b, p, q, nn)
  dro <- rd_cote(x[!g], y[!g], c, h, b, p, q, nn)
  tau <- dro$tau - gau$tau
  tau_bc <- dro$tau_bc - gau$tau_bc
  se <- sqrt(gau$v_cl + dro$v_cl)
  se_rb <- sqrt(gau$v_rb + dro$v_rb)
  z <- stats::qnorm(1 - (1 - niveau) / 2)
  structure(list(tau = tau, tau_bc = tau_bc, se = se, se_rb = se_rb,
                 ci = tau + c(-1, 1) * z * se, ci_rb = tau_bc + c(-1, 1) * z * se_rb,
                 pvalue = 2 * stats::pnorm(-abs(tau_bc / se_rb)),
                 h = h, b = b, n = c(gau$n_h, dro$n_h), c = c, p = p, q = q),
            class = "rd_robust")
}

print.rd_robust <- function(x, ...) {
  cat(sprintf("RD au seuil %g : h = %.3f, b = %.3f, n = (%d, %d)\n", x$c, x$h, x$b, x$n[1], x$n[2]))
  cat(sprintf("  classique : %.4f (se %.4f)\n", x$tau, x$se))
  cat(sprintf("  corrige   : %.4f (se robuste %.4f), IC robuste [%.3f, %.3f], p = %.4f\n",
              x$tau_bc, x$se_rb, x$ci_rb[1], x$ci_rb[2], x$pvalue))
  invisible(x)
}

# Constantes de variance et de biais pour la fenetre optimale (une etape, un cote).
# o : ordre du polynome, nu : derivee visee, oB : ordre du polynome qui estime le biais,
# hV : fenetre de la variance, hB : fenetre de l'estimation du biais
rd_const <- function(x, y, c, o, nu, oB, hV, hB, nn) {
  dv <- abs(x - c) <= hV
  xv <- x[dv]
  yv <- y[dv]
  s2v <- rd_nn_res(xv, yv, nn)^2
  av <- rd_poids(xv, c, hV, o, nu)
  V <- hV^(2 * nu + 1) * sum(av$a^2 * s2v[av$idx])
  bc <- sum(av$a * (xv[av$idx] - c)^(o + 1)) / hV^(o + 1 - nu)
  db <- abs(x - c) <= hB
  xb <- x[db]
  yb <- y[db]
  s2b <- rd_nn_res(xb, yb, nn)^2
  ab <- rd_poids(xb, c, hB, oB, o + 1)
  beta <- sum(ab$a * yb[ab$idx])
  list(V = V, B = bc * beta, R = 3 * bc^2 * sum(ab$a^2 * s2b[ab$idx]))
}

# Fenetres h et b minimisant l'erreur quadratique moyenne (une seule fenetre pour les deux cotes)
rd_bandwidth <- function(y, x, c = 0, p = 1, q = p + 1, nn = 3, regul = 1) {
  ok <- !is.na(y) & !is.na(x)
  y <- y[ok]
  x <- x[ok]
  x_sd <- stats::sd(x)
  x <- x / x_sd
  c <- c / x_sd
  xl <- x[x < c]; yl <- y[x < c]
  xr <- x[x >= c]; yr <- y[x >= c]
  M <- length(unique(xl)) + length(unique(xr))
  c_bw <- 2.576 * min(1, stats::IQR(x) / 1.349) * M^(-1 / 5)
  etape <- function(o, nu, oB, hBl, hBr, regul) {
    L <- rd_const(xl, yl, c, o, nu, oB, c_bw, hBl, nn)
    R <- rd_const(xr, yr, c, o, nu, oB, c_bw, hBr, nn)
    B <- R$B - L$B
    ((2 * nu + 1) * (L$V + R$V) /
       (2 * (o + 1 - nu) * (B^2 + regul * (L$R + R$R))))^(1 / (2 * o + 3))
  }
  d <- etape(q + 1, q + 1, q + 2, c - min(x), max(x) - c, 0)
  b <- etape(q, p + 1, q + 1, d, d, regul)
  h <- etape(p, 0, q, b, b, regul)
  c(h = x_sd * h, b = x_sd * b)
}

# Effet estime a de faux seuils, en n'utilisant que les observations du meme cote du vrai seuil
rd_placebo <- function(y, x, c = 0, placebos, ...) {
  ok <- !is.na(y) & !is.na(x)
  y <- y[ok]
  x <- x[ok]
  lignes <- lapply(placebos, function(cp) {
    sel <- if (cp < c) x < c else x >= c
    r <- tryCatch(rd_robust(y[sel], x[sel], cp, ...), error = function(e) NULL)
    if (is.null(r)) {
      return(data.frame(placebo = cp, tau = NA_real_, tau_bc = NA_real_, se_rb = NA_real_,
                        pvalue = NA_real_, h = NA_real_))
    }
    data.frame(placebo = cp, tau = r$tau, tau_bc = r$tau_bc, se_rb = r$se_rb,
               pvalue = r$pvalue, h = r$h)
  })
  do.call(rbind, lignes)
}

# Test de manipulation : binomiale sur les effectifs de part et d'autre du seuil
rd_manipulation <- function(x, c = 0, h) {
  g <- sum(x >= c - h & x < c)
  d <- sum(x >= c & x <= c + h)
  list(gauche = g, droite = d, pvalue = stats::binom.test(c(g, d), p = 0.5)$p.value)
}

plot_rd <- function(y, x, c = 0, main = "Discontinuite", nbins = 20) {
  ok <- !is.na(y) & !is.na(x)
  y <- y[ok]
  x <- x[ok]
  r <- rd_robust(y, x, c)
  classes <- function(sel) {
    q <- unique(stats::quantile(x[sel], seq(0, 1, length.out = nbins + 1)))
    g <- cut(x[sel], q, include.lowest = TRUE)
    data.frame(x = as.vector(tapply(x[sel], g, mean)), y = as.vector(tapply(y[sel], g, mean)))
  }
  pts <- rbind(classes(x < c), classes(x >= c))
  graphics::plot(pts$x, pts$y, pch = 19, main = main, xlab = "variable d'assignation",
                 ylab = "resultat")
  graphics::abline(v = c, lty = 3)
  for (cote in list(x < c, x >= c)) {
    sel <- cote & abs(x - c) <= r$h
    dd <- data.frame(y = y[sel], u = x[sel] - c)
    f <- stats::lm(y ~ u, data = dd, weights = 1 - abs(dd$u) / r$h)
    xs <- range(x[sel])
    graphics::lines(xs, stats::coef(f)[1] + stats::coef(f)[2] * (xs - c), lwd = 2)
  }
}
