mpdta <- lire("data", "mpdta.csv")

# Reference : package R did (tests/references/generer.R), sans bootstrap
configs <- list(
  jamais_sans = list(cg = "nevertreated", f = ~1, base = "varying", m = "dr"),
  pas_encore_sans = list(cg = "notyettreated", f = ~1, base = "varying", m = "dr"),
  jamais_cov = list(cg = "nevertreated", f = ~lpop, base = "varying", m = "dr"),
  pas_encore_cov = list(cg = "notyettreated", f = ~lpop, base = "varying", m = "dr"),
  jamais_universel = list(cg = "nevertreated", f = ~1, base = "universal", m = "dr"),
  jamais_cov_reg = list(cg = "nevertreated", f = ~lpop, base = "varying", m = "reg"),
  jamais_cov_ipw = list(cg = "nevertreated", f = ~lpop, base = "varying", m = "ipw"),
  pas_encore_cov_reg = list(cg = "notyettreated", f = ~lpop, base = "varying", m = "reg")
)

for (nom in names(configs)) {
  test_that(paste("ATT(g, t) et agregations identiques a did :", nom), {
    skip_if_not(ref_existe(paste0("did_", nom, "_attgt.csv")), "references absentes")
    cfg <- configs[[nom]]
    res <- att_gt(mpdta, "lemp", "year", "countyreal", "first.treat", xformla = cfg$f,
                  control_group = cfg$cg, est_method = cfg$m, base_period = cfg$base)
    ref <- lire("tests", "references", paste0("did_", nom, "_attgt.csv"))
    expect_equal(res$group, ref$g)
    expect_equal(res$t, ref$t)
    expect_equal(res$att, ref$att, tolerance = 1e-8)
    expect_equal(res$se, ref$se, tolerance = 1e-6)
    agg <- lire("tests", "references", paste0("did_", nom, "_agg.csv"))
    for (ty in c("simple", "dynamic", "group", "calendar")) {
      a <- aggte(res, ty)
      sous <- agg[agg$type == ty, ]
      global <- sous[sous$global == 1, ]
      expect_equal(a$overall_att, global$att, tolerance = 1e-7)
      expect_equal(a$overall_se, global$se, tolerance = 1e-5)
      if (!is.null(a$egt)) {
        detail <- sous[sous$global == 0, ]
        expect_equal(a$egt, detail$indice)
        expect_equal(a$att_egt, detail$att, tolerance = 1e-7)
        expect_equal(a$se_egt, detail$se, tolerance = 1e-5)
      }
    }
  })
}

# Panel simule avec adoption echelonnee et effets heterogenes : trois cohortes (traitees en 3 et en 6,
# jamais traitees), l'effet croit avec la duree d'exposition et differe selon la cohorte.
simuler_panel <- function(graine = 1, n_par_groupe = 500, nT = 8) {
  set.seed(graine)
  n <- 3 * n_par_groupe
  g <- rep(c(3, 6, 0), each = n_par_groupe)
  id <- rep(seq_len(n), each = nT)
  t <- rep(seq_len(nT), n)
  gi <- g[id]
  delta <- c(`3` = 1.0, `6` = 0.2, `0` = 0)[as.character(gi)]
  tau <- ifelse(gi > 0 & t >= gi, delta * (t - gi + 1), 0)
  a <- rnorm(n)[id]
  y <- a + 0.3 * t + tau + rnorm(n * nT)
  data.frame(id = id, t = t, g = gi, y = y, d = as.numeric(gi > 0 & t >= gi))
}

# Effet vrai de chaque cellule post-traitement, puis moyenne ponderee par la taille des cohortes
test_that("Callaway-Sant'Anna retrouve l'effet vrai, pas le DiD a effets fixes bi-directionnels", {
  d <- simuler_panel()
  res <- att_gt(d, "y", "t", "id", "g")
  vrai <- ifelse(res$post == 1, c(`3` = 1.0, `6` = 0.2)[as.character(res$group)] * (res$t - res$group + 1), 0)
  expect_equal(res$att, vrai, tolerance = 0.25, scale = 1)
  simple <- aggte(res, "simple")
  verite <- sum(vrai[res$post == 1]) / sum(res$post == 1)
  expect_lt(abs(simple$overall_att - verite), 3 * simple$overall_se)
  twfe <- twfe_did(d, "y", "id", "t", "d")
  expect_gt(abs(twfe$coef - verite), 0.25)
  expect_gt(abs(twfe$coef - verite), 3 * twfe$se)
})

test_that("les cellules avant traitement sont proches de zero (tendances paralleles)", {
  res <- att_gt(simuler_panel(2), "y", "t", "id", "g", base_period = "universal")
  avant <- res$post == 0 & res$att != 0
  expect_lt(max(abs(res$att[avant] / res$se[avant])), 4)
})

test_that("DiD 2x2 : une seule cohorte donne la difference de differences de moyennes", {
  d <- simuler_panel(3)
  d <- d[d$g %in% c(3, 0) & d$t %in% c(2, 3), ]
  res <- att_gt(d, "y", "t", "id", "g", est_method = "reg")
  m <- tapply(d$y, list(d$g, d$t), mean)
  expect_equal(res$att[res$t == 3], unname((m["3", "3"] - m["3", "2"]) - (m["0", "3"] - m["0", "2"])), tolerance = 1e-10)
})

test_that("un panel desequilibre est refuse", {
  d <- simuler_panel(4)
  expect_error(att_gt(d[-1, ], "y", "t", "id", "g"), "desequilibre")
})
