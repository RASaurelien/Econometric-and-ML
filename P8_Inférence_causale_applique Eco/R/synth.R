# Controle synthetique (Abadie et Gardeazabal, 2003 ; Abadie, Diamond et Hainmueller, 2010)

synth_prep <- function(data, unit, time, outcome, treated, controls, traitement, time_optimize,
                       predictors = NULL, predictor_periods = NULL, special = NULL, noms = NULL) {
  args <- list(data = data, unit = unit, time = time, outcome = outcome, treated = treated,
               controls = controls, traitement = traitement, time_optimize = time_optimize,
               predictors = predictors, predictor_periods = predictor_periods,
               special = special, noms = noms)
  ids <- c(treated, controls)
  moy <- function(v, periodes) {
    vapply(ids, function(i) {
      mean(data[[v]][data[[unit]] == i & data[[time]] %in% periodes], na.rm = TRUE)
    }, numeric(1))
  }
  lignes <- list()
  etiq <- character(0)
  for (v in predictors) {
    lignes[[length(lignes) + 1]] <- moy(v, predictor_periods)
    etiq <- c(etiq, v)
  }
  for (s in special) {
    lignes[[length(lignes) + 1]] <- moy(s$variable, s$periods)
    etiq <- c(etiq, paste0(s$variable, ".", min(s$periods), "-", max(s$periods)))
  }
  X <- do.call(rbind, lignes)
  rownames(X) <- etiq
  colnames(X) <- ids

  annees <- sort(unique(data[[time]]))
  Y <- vapply(ids, function(i) {
    d <- data[data[[unit]] == i, ]
    d[[outcome]][match(annees, d[[time]])]
  }, numeric(length(annees)))
  rownames(Y) <- annees
  colnames(Y) <- ids
  Z <- Y[as.character(time_optimize), , drop = FALSE]

  etiquettes <- as.character(controls)
  if (!is.null(noms)) {
    etiquettes <- as.character(data[[noms]][match(controls, data[[unit]])])
  }
  structure(list(X1 = X[, 1, drop = FALSE], X0 = X[, -1, drop = FALSE],
                 Z1 = Z[, 1], Z0 = Z[, -1, drop = FALSE],
                 Y1 = Y[, 1], Y0 = Y[, -1, drop = FALSE], annees = annees,
                 treated = treated, controls = controls, etiquettes = etiquettes,
                 traitement = traitement, args = args), class = "synth_prep")
}

# Les predicteurs sont divises par leur ecart-type sur toutes les unites, comme dans Synth
synth_echelle <- function(dp) {
  s <- apply(cbind(dp$X1, dp$X0), 1, stats::sd)
  s[is.na(s) | s == 0] <- 1
  list(X1 = dp$X1 / s, X0 = dp$X0 / s)
}

# Poids W pour une matrice V diagonale donnee (programme quadratique sous contraintes)
synth_qp <- function(X0, X1, v) {
  J <- ncol(X0)
  D <- crossprod(X0, v * X0)
  d <- drop(crossprod(X0, v * X1))
  D <- (D + t(D)) / 2
  D <- D + diag(1e-8 * mean(diag(D)), J)
  sol <- quadprog::solve.QP(D, d, cbind(1, diag(J)), c(1, rep(0, J)), meq = 1)
  w <- pmax(sol$solution, 0)
  list(w = w / sum(w), D = D, d = d)
}

synth <- function(dp, n_depart = 6) {
  esc <- synth_echelle(dp)
  k <- nrow(esc$X0)
  poids_v <- function(theta) {
    v <- exp(theta - max(theta))
    v / sum(v)
  }
  perte <- function(theta) {
    w <- synth_qp(esc$X0, esc$X1, poids_v(theta))$w
    mean((dp$Z1 - drop(dp$Z0 %*% w))^2)
  }
  if (k == 1) {
    best <- list(par = 0, value = perte(0))
  } else {
    best <- NULL
    for (s in seq_len(n_depart)) {
      theta0 <- if (s == 1) rep(0, k) else 2 * sin(seq_len(k) * (s - 1) * 1.7 + s)
      o <- stats::optim(theta0, perte, method = "Nelder-Mead",
                        control = list(maxit = 600, reltol = 1e-10))
      if (is.null(best) || o$value < best$value) best <- o
    }
  }
  v <- poids_v(best$par)
  w <- synth_qp(esc$X0, esc$X1, v)$w
  structure(list(W = stats::setNames(w, dp$etiquettes), V = stats::setNames(v, rownames(dp$X0)),
                 perte = best$value, dp = dp), class = "synth_fit")
}

print.synth_fit <- function(x, ...) {
  cat("Controle synthetique, EQM avant traitement :", format(x$perte, digits = 4), "\n")
  w <- sort(x$W[x$W > 0.001], decreasing = TRUE)
  cat("Poids :\n")
  print(round(w, 3))
  invisible(x)
}

synth_ecart <- function(dp, W) {
  syn <- drop(dp$Y0 %*% W)
  data.frame(annee = dp$annees, traite = dp$Y1, synthetique = syn, ecart = dp$Y1 - syn)
}

# Placebos dans l'espace : chaque controle joue le role de l'unite traitee
synth_placebo <- function(fit, n_depart = 2) {
  dp <- fit$dp
  a <- dp$args
  rapport <- function(d, w) {
    e <- synth_ecart(d, w)
    avant <- e$annee < d$traitement
    pre <- sqrt(mean(e$ecart[avant]^2))
    post <- sqrt(mean(e$ecart[!avant]^2))
    c(pre = pre, post = post, rapport = post / pre)
  }
  unites <- c(a$treated, a$controls)
  lignes <- lapply(unites, function(u) {
    if (u == a$treated) {
      r <- rapport(dp, fit$W)
    } else {
      d <- do.call(synth_prep, utils::modifyList(a, list(treated = u,
                                                         controls = setdiff(a$controls, u))))
      r <- rapport(d, synth(d, n_depart = n_depart)$W)
    }
    data.frame(unite = u, rmspe_avant = r[["pre"]], rmspe_apres = r[["post"]],
               rapport = r[["rapport"]])
  })
  tab <- do.call(rbind, lignes)
  list(table = tab, rapport_traite = tab$rapport[1], pvalue = mean(tab$rapport >= tab$rapport[1]))
}

plot_synth <- function(fit, main = "Controle synthetique") {
  dp <- fit$dp
  e <- synth_ecart(dp, fit$W)
  graphics::plot(e$annee, e$traite, type = "l", lwd = 2, ylim = range(e$traite, e$synthetique, na.rm = TRUE),
                 xlab = "", ylab = dp$args$outcome, main = main)
  graphics::lines(e$annee, e$synthetique, lwd = 2, lty = 2)
  graphics::abline(v = dp$traitement, lty = 3)
  graphics::legend("topleft", c("traite", "synthetique"), lty = c(1, 2), lwd = 2, bty = "n")
}
