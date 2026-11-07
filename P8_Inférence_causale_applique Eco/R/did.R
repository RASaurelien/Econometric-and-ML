# Differences de differences avec adoption echelonnee (Callaway et Sant'Anna, 2021)

# ATT et fonction d'influence pour une cellule (g, t), panel de deux periodes.
# dy : variation de l'outcome, D : 1 si cohorte g, 0 si controle, X : covariables (avec constante)
est_did <- function(dy, D, X, methode) {
  n <- length(dy)
  if (methode != "ipw") {
    Xc <- X[D == 0, , drop = FALSE]
    beta <- solve(crossprod(Xc), crossprod(Xc, dy[D == 0]))
    or <- drop(X %*% beta)
    inv <- solve(crossprod(X * (1 - D), X) / n)
    lin_ols <- (X * ((1 - D) * (dy - or))) %*% inv
  }
  if (methode != "reg") {
    fit <- logit(X, D)
    ps <- pmin(fit$p, 1 - 1e-6)
    lin_ps <- (X * (D - ps)) %*% (fit$vcov * n)
    wc <- ps * (1 - D) / (1 - ps)
  }
  mt <- mean(D)
  if (methode == "reg") {
    eta_t <- mean(D * dy) / mt
    eta_c <- mean(D * or) / mt
    inf_t <- (D * dy - D * eta_t) / mt
    inf_c <- (D * or - D * eta_c + lin_ols %*% colMeans(D * X)) / mt
    att <- eta_t - eta_c
  } else if (methode == "ipw") {
    mc <- mean(wc)
    eta_t <- mean(D * dy) / mt
    eta_c <- mean(wc * dy) / mc
    inf_t <- (D * dy - D * eta_t) / mt
    inf_c <- (wc * dy - wc * eta_c + lin_ps %*% colMeans(wc * (dy - eta_c) * X)) / mc
    att <- eta_t - eta_c
  } else {
    mc <- mean(wc)
    e_t <- D * (dy - or) / mt
    e_c <- wc * (dy - or) / mc
    att_t <- mean(e_t)
    att_c <- mean(e_c)
    inf_t <- e_t - D * att_t / mt - lin_ols %*% (colMeans(D * X) / mt)
    inf_c <- e_c - wc * att_c / mc + lin_ps %*% (colMeans(wc * (dy - or - att_c) * X) / mc) -
      lin_ols %*% (colMeans(wc * X) / mc)
    att <- att_t - att_c
  }
  list(att = att, inf = drop(inf_t - inf_c))
}

att_gt <- function(data, yname, tname, idname, gname, xformla = NULL,
                   control_group = c("nevertreated", "notyettreated"),
                   est_method = c("dr", "ipw", "reg"),
                   base_period = c("varying", "universal")) {
  control_group <- match.arg(control_group)
  est_method <- match.arg(est_method)
  base_period <- match.arg(base_period)
  data <- data[order(data[[idname]], data[[tname]]), ]
  temps <- sort(unique(data[[tname]]))
  Tn <- length(temps)
  n <- length(unique(data[[idname]]))
  if (nrow(data) != n * Tn || any(table(data[[idname]]) != Tn) ||
      anyDuplicated(data[, c(idname, tname)]) > 0) {
    stop("panel desequilibre : chaque individu doit etre observe une fois a chaque date")
  }
  Y <- matrix(data[[yname]], nrow = n, ncol = Tn, byrow = TRUE)
  G <- data[[gname]][(seq_len(n) - 1) * Tn + 1]
  G[is.na(G) | is.infinite(G)] <- 0
  glist <- sort(unique(G[G > 0 & G > temps[1]]))
  Xl <- NULL
  if (!is.null(xformla)) {
    Xl <- lapply(temps, function(s) stats::model.matrix(xformla, data[data[[tname]] == s, ]))
  }

  grp <- tt <- att <- post <- numeric(0)
  infs <- list()
  for (g in glist) {
    pre <- max(which(temps < g))
    ks <- if (base_period == "universal") seq_len(Tn) else 2:Tn
    for (k in ks) {
      base <- if (base_period == "universal") pre else if (temps[k] < g) k - 1 else pre
      if (k == base) {
        a <- 0
        inf <- numeric(n)
      } else {
        later <- max(temps[k], temps[base])
        traite <- G == g
        ctrl <- if (control_group == "nevertreated") G == 0 else (G == 0 | (G > later & G != g))
        if (sum(traite) == 0 || sum(ctrl) == 0) next
        sub <- traite | ctrl
        D <- as.numeric(traite[sub])
        dy <- Y[sub, k] - Y[sub, base]
        X <- if (is.null(Xl)) matrix(1, sum(sub), 1) else Xl[[min(k, base)]][sub, , drop = FALSE]
        e <- est_did(dy, D, X, est_method)
        a <- e$att
        inf <- numeric(n)
        inf[sub] <- e$inf * n / sum(sub)
      }
      grp <- c(grp, g)
      tt <- c(tt, temps[k])
      att <- c(att, a)
      post <- c(post, as.numeric(g <= temps[k]))
      infs[[length(infs) + 1]] <- inf
    }
  }
  inf <- do.call(cbind, infs)
  se <- sqrt(colMeans(inf^2) / n)
  se[se <= sqrt(.Machine$double.eps) * 10] <- NA
  structure(list(group = grp, t = tt, att = att, se = se, post = post, inf = inf, n = n, G = G,
                 glist = glist, tlist = temps, control_group = control_group,
                 est_method = est_method, base_period = base_period), class = "att_gt")
}

print.att_gt <- function(x, ...) {
  cat("ATT(g, t) :", x$control_group, "/", x$est_method, "/", x$base_period, "\n")
  print(data.frame(groupe = x$group, date = x$t, att = round(x$att, 4), se = round(x$se, 4)),
        row.names = FALSE)
  invisible(x)
}

# Agregations des ATT(g, t) : "simple", "dynamic", "group" ou "calendar"
aggte <- function(res, type = c("simple", "dynamic", "group", "calendar")) {
  type <- match.arg(type)
  grp <- res$group; tt <- res$t; att <- res$att; inf <- res$inf
  n <- res$n; G <- res$G
  glist <- sort(unique(grp))
  pg <- vapply(glist, function(g) mean(G == g), numeric(1))
  pgc <- pg[match(grp, glist)]

  se_vec <- function(v) {
    s <- se_influence(v)
    if (s <= sqrt(.Machine$double.eps) * 10) NA_real_ else s
  }
  # influence des poids estimes (tailles de cohortes)
  wif <- function(gs, ps) {
    K <- length(gs)
    ind <- vapply(gs, function(g) as.numeric(G == g), numeric(n))
    dev <- ind - matrix(ps, n, K, byrow = TRUE)
    s <- sum(ps)
    dev / s - rowSums(dev) %o% (ps / s^2)
  }
  combine <- function(ks, w, avec_wif = TRUE) {
    f <- inf[, ks, drop = FALSE] %*% w
    if (avec_wif) f <- f + wif(grp[ks], pgc[ks]) %*% att[ks]
    drop(f)
  }
  fin <- function(overall, ifo, egt = NULL, att_egt = NULL, se_egt = NULL) {
    structure(list(type = type, overall_att = overall, overall_se = se_vec(ifo),
                   egt = egt, att_egt = att_egt, se_egt = se_egt), class = "aggte")
  }

  if (type == "simple") {
    ks <- which(grp <= tt)
    w <- pgc[ks] / sum(pgc[ks])
    return(fin(sum(att[ks] * w), combine(ks, w)))
  }
  if (type == "dynamic") {
    eseq <- sort(unique(tt - grp))
    parts <- lapply(eseq, function(e) {
      ks <- which(tt - grp == e)
      w <- pgc[ks] / sum(pgc[ks])
      list(att = sum(att[ks] * w), f = combine(ks, w))
    })
    att_e <- vapply(parts, function(p) p$att, numeric(1))
    f_e <- vapply(parts, function(p) p$f, numeric(n))
    pos <- eseq >= 0
    return(fin(mean(att_e[pos]), rowMeans(f_e[, pos, drop = FALSE]),
               eseq, att_e, apply(f_e, 2, se_vec)))
  }
  if (type == "group") {
    parts <- lapply(glist, function(g) {
      ks <- which(grp == g & tt >= g)
      list(att = mean(att[ks]), f = combine(ks, rep(1 / length(ks), length(ks)), FALSE))
    })
    att_g <- vapply(parts, function(p) p$att, numeric(1))
    f_g <- vapply(parts, function(p) p$f, numeric(n))
    wg <- pg / sum(pg)
    ifo <- drop(f_g %*% wg) + drop(wif(glist, pg) %*% att_g)
    return(fin(sum(att_g * wg), ifo, glist, att_g, apply(f_g, 2, se_vec)))
  }
  tl <- res$tlist[res$tlist >= min(grp)]
  parts <- lapply(tl, function(s) {
    ks <- which(tt == s & grp <= tt)
    w <- pgc[ks] / sum(pgc[ks])
    list(att = sum(att[ks] * w), f = combine(ks, w))
  })
  att_t <- vapply(parts, function(p) p$att, numeric(1))
  f_t <- vapply(parts, function(p) p$f, numeric(n))
  fin(mean(att_t), rowMeans(f_t), tl, att_t, apply(f_t, 2, se_vec))
}

print.aggte <- function(x, ...) {
  cat(sprintf("Agregation %s : ATT = %.4f (se %.4f)\n", x$type, x$overall_att, x$overall_se))
  if (!is.null(x$egt)) {
    print(data.frame(indice = x$egt, att = round(x$att_egt, 4), se = round(x$se_egt, 4)),
          row.names = FALSE)
  }
  invisible(x)
}

# DiD a effets fixes bi-directionnels (panel equilibre), ecart-type groupe par individu
twfe_did <- function(data, yname, idname, tname, dname) {
  id <- data[[idname]]; t <- data[[tname]]
  dem <- function(v) v - stats::ave(v, id) - stats::ave(v, t) + mean(v)
  yt <- dem(data[[yname]])
  dt <- dem(data[[dname]])
  b <- sum(dt * yt) / sum(dt^2)
  s <- tapply(dt * (yt - b * dt), id, sum)
  Gn <- length(s)
  list(coef = b, se = sqrt(Gn / (Gn - 1) * sum(s^2)) / sum(dt^2))
}

plot_event_study <- function(agg, main = "Etude d'evenement", niveau = 0.95) {
  z <- stats::qnorm(1 - (1 - niveau) / 2)
  se <- agg$se_egt
  se[is.na(se)] <- 0
  bas <- agg$att_egt - z * se
  haut <- agg$att_egt + z * se
  graphics::plot(agg$egt, agg$att_egt, pch = 19, ylim = range(bas, haut, 0), main = main,
                 xlab = "temps depuis le traitement", ylab = "effet")
  graphics::segments(agg$egt, bas, agg$egt, haut)
  graphics::abline(h = 0, lty = 2)
  graphics::abline(v = -0.5, lty = 3)
}
