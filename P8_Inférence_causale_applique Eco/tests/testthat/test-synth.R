speciaux_basque <- c(
  list(list(variable = "gdpcap", periods = 1960:1969)),
  lapply(c("sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
           "sec.services.venta", "sec.services.nonventa"),
         function(v) list(variable = v, periods = seq(1961, 1969, 2))),
  list(list(variable = "popdens", periods = 1969))
)

prep_basque <- function() {
  basque <- lire("data", "basque.csv")
  synth_prep(basque, "regionno", "year", "gdpcap", 17, c(2:16, 18), traitement = 1970,
             time_optimize = 1960:1969,
             predictors = c("school.illit", "school.prim", "school.med", "school.high",
                            "school.post.high", "invest"),
             predictor_periods = 1964:1969, special = speciaux_basque)
}

# Modele a facteurs : l'unite traitee est une combinaison convexe de deux controles
simuler_synth <- function(effet, graine = 1, N = 20, Tn = 30, T0 = 20) {
  set.seed(graine)
  F <- matrix(rnorm(Tn * 3), Tn, 3)
  mu <- matrix(runif(N * 3), N, 3)
  mu[1, ] <- 0.5 * mu[2, ] + 0.5 * mu[3, ]
  y <- F %*% t(mu) + matrix(rnorm(Tn * N, sd = 0.1), Tn, N)
  y[(T0 + 1):Tn, 1] <- y[(T0 + 1):Tn, 1] + effet
  data.frame(u = rep(seq_len(N), each = Tn), t = rep(seq_len(Tn), N), y = as.vector(y))
}

prep_sim <- function(d, T0 = 20) {
  blocs <- list(1:5, 6:10, 11:15, 16:20)
  synth_prep(d, "u", "t", "y", 1, 2:20, traitement = T0 + 1, time_optimize = 1:T0,
             special = lapply(blocs, function(p) list(variable = "y", periods = p)))
}

test_that("les matrices de predicteurs sont identiques a celles de Synth", {
  skip_if_not(ref_existe("synth_basque_X.csv"), "references absentes")
  dp <- prep_basque()
  ref <- lire("tests", "references", "synth_basque_X.csv")
  expect_equal(unname(cbind(dp$X1, dp$X0)), unname(as.matrix(ref[, -1])), tolerance = 1e-8)
})

test_that("avec la meme matrice V, les poids sont ceux de Synth", {
  skip_if_not(ref_existe("synth_basque_poids.csv"), "references absentes")
  dp <- prep_basque()
  esc <- interne("synth_echelle")(dp)
  v <- lire("tests", "references", "synth_basque_v.csv")$v
  w_ref <- lire("tests", "references", "synth_basque_poids.csv")$w_v
  w <- interne("synth_qp")(esc$X0, esc$X1, v)$w
  expect_lt(max(abs(w - w_ref)), 2e-3)
})

test_that("les poids verifient les conditions de Kuhn-Tucker du programme quadratique", {
  skip_if_not(file.exists(file.path(racine, "data", "basque.csv")), "donnees absentes")
  dp <- prep_basque()
  esc <- interne("synth_echelle")(dp)
  v <- rep(1 / nrow(esc$X0), nrow(esc$X0))
  q <- interne("synth_qp")(esc$X0, esc$X1, v)
  expect_equal(sum(q$w), 1)
  expect_true(all(q$w >= 0))
  grad <- drop(q$D %*% q$w - q$d)
  actifs <- q$w > 1e-6
  lambda <- mean(grad[actifs])
  expect_lt(max(abs(grad[actifs] - lambda)), 1e-4)
  expect_true(all(grad[!actifs] >= lambda - 1e-4))
})

test_that("le choix de V fait au moins aussi bien que Synth sur la perte avant traitement", {
  skip_if_not(ref_existe("synth_basque_poids.csv"), "references absentes")
  dp <- prep_basque()
  w_opt <- lire("tests", "references", "synth_basque_poids.csv")$w_opt
  perte_ref <- mean((dp$Z1 - drop(dp$Z0 %*% w_opt))^2)
  fit <- synth(dp)
  expect_lt(fit$perte, perte_ref * 1.05 + 1e-8)
})

test_that("sur un modele a facteurs simule, l'effet vrai est retrouve", {
  dp <- prep_sim(simuler_synth(effet = 2))
  fit <- synth(dp)
  e <- synth_ecart(dp, fit$W)
  expect_lt(abs(mean(e$ecart[e$annee > 20]) - 2), 0.5)
  expect_lt(max(abs(e$ecart[e$annee <= 20])), 0.5)
})

test_that("les placebos donnent une p-value basse avec un effet et pas sans effet", {
  avec <- synth(prep_sim(simuler_synth(effet = 5)))
  sans <- synth(prep_sim(simuler_synth(effet = 0)))
  expect_lte(synth_placebo(avec)$pvalue, 0.1)
  expect_gt(synth_placebo(sans)$pvalue, 0.05)
})
