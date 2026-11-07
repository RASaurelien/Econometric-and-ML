senat <- lire("data", "rdsenate.csv")
senat <- senat[!is.na(senat$vote), ]
ref <- if (ref_existe("rd_senat.csv")) lire("tests", "references", "rd_senat.csv")

test_that("estimation, variances et effectifs identiques a rdrobust pour des fenetres donnees", {
  skip_if_not(!is.null(ref), "references absentes")
  for (i in seq_len(nrow(ref))) {
    r <- rd_robust(senat$vote, senat$margin, 0, h = ref$h[i], b = ref$b[i])
    expect_equal(r$tau, ref$tau[i], tolerance = 1e-9)
    expect_equal(r$tau_bc, ref$tau_bc[i], tolerance = 1e-9)
    expect_equal(r$se, ref$se[i], tolerance = 1e-9)
    expect_equal(r$se_rb, ref$se_rb[i], tolerance = 1e-9)
    expect_equal(unname(r$n), c(ref$n_gauche[i], ref$n_droite[i]))
  }
})

test_that("la fenetre optimale redonne celle de rdrobust (mserd)", {
  skip_if_not(!is.null(ref), "references absentes")
  bw <- rd_bandwidth(senat$vote, senat$margin)
  expect_equal(unname(bw), c(17.75439722, 28.02808715), tolerance = 1e-7)
  r <- rd_robust(senat$vote, senat$margin, 0)
  expect_equal(r$h, 17.75439722, tolerance = 1e-7)
  expect_equal(r$tau_bc, ref$tau_bc[3], tolerance = 1e-6)
})

simuler_rd <- function(graine, n = 3000, saut = 1) {
  set.seed(graine)
  x <- runif(n, -1, 1)
  data.frame(x = x, y = 0.5 + x + 0.8 * x^2 + saut * (x >= 0) + rnorm(n, sd = 0.5))
}

test_that("le saut vrai est retrouve et l'intervalle robuste le contient", {
  d <- simuler_rd(1)
  r <- rd_robust(d$y, d$x, 0)
  expect_lt(abs(r$tau_bc - 1), 3 * r$se_rb)
  expect_true(r$ci_rb[1] < 1 && 1 < r$ci_rb[2])
  expect_lt(r$pvalue, 1e-6)
})

test_that("couverture de l'intervalle robuste proche de 95 %", {
  couvert <- vapply(1:150, function(s) {
    d <- simuler_rd(100 + s, n = 1500)
    r <- rd_robust(d$y, d$x, 0)
    r$ci_rb[1] < 1 && 1 < r$ci_rb[2]
  }, logical(1))
  expect_gt(mean(couvert), 0.90)
  expect_lt(mean(couvert), 0.995)
})

test_that("sans saut, l'effet estime est nul et les placebos ne rejettent pas", {
  d <- simuler_rd(2, saut = 0)
  expect_gt(rd_robust(d$y, d$x, 0)$pvalue, 0.01)
  pl <- rd_placebo(d$y, d$x, 0, placebos = c(-0.6, -0.4, 0.4, 0.6))
  expect_true(all(pl$pvalue > 0.01))
})

test_that("la correction de biais reduit le biais avec une forte courbure", {
  d <- simuler_rd(3, n = 6000)
  d$y <- d$y + 3 * d$x^2 * (d$x >= 0)
  r <- rd_robust(d$y, d$x, 0, h = 0.5, b = 0.5)
  expect_lt(abs(r$tau_bc - 1), abs(r$tau - 1))
})

test_that("test de manipulation : accepte sans tri, rejette avec un tri au seuil", {
  set.seed(5)
  x <- runif(4000, -1, 1)
  expect_gt(rd_manipulation(x, 0, 0.1)$pvalue, 0.01)
  tri <- c(x, runif(200, 0, 0.05))
  expect_lt(rd_manipulation(tri, 0, 0.1)$pvalue, 1e-6)
})

test_that("residus par plus proches voisins avec des ex aequo", {
  x <- c(1, 1, 2, 3, 3, 3, 4)
  y <- c(2, 4, 3, 1, 5, 9, 0)
  r <- interne("rd_nn_res")(x, y, 2)
  expect_length(r, 7)
  expect_true(all(is.finite(r)))
})
