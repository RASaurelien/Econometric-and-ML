# A lancer depuis la racine du projet (de preference apres R CMD INSTALL .)
if (requireNamespace("minicausal", quietly = TRUE)) {
  library(minicausal)
} else {
  for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) source(f)
}
set.seed(1)

# 1. Differences de differences : salaire minimum et emploi des jeunes (donnees mpdta du package did)
cat("DIFFERENCES DE DIFFERENCES (Callaway et Sant'Anna), mpdta\n")
mpdta <- read.csv("data/mpdta.csv")
res <- att_gt(mpdta, "lemp", "year", "countyreal", "first.treat", control_group = "nevertreated")
print(res)
cat("\n")
simple <- aggte(res, "simple")
dyn <- aggte(res, "dynamic")
print(simple)
print(dyn)
mpdta$d <- as.numeric(mpdta$first.treat > 0 & mpdta$year >= mpdta$first.treat)
tw <- twfe_did(mpdta, "lemp", "countyreal", "year", "d")
cat(sprintf("DiD a effets fixes bi-directionnels : %.4f (se %.4f)\n", tw$coef, tw$se))

# Adoption echelonnee simulee avec effets heterogenes : le DiD classique se trompe
n_g <- 600; nT <- 8
g <- rep(c(3, 6, 0), each = n_g)
id <- rep(seq_along(g), each = nT); t <- rep(1:nT, length(g)); gi <- g[id]
tau <- ifelse(gi > 0 & t >= gi, c(`3` = 1.0, `6` = 0.2, `0` = 0)[as.character(gi)] * (t - gi + 1), 0)
sim <- data.frame(id = id, t = t, g = gi, d = as.numeric(gi > 0 & t >= gi),
                  y = rnorm(length(g))[id] + 0.3 * t + tau + rnorm(length(id)))
rs <- att_gt(sim, "y", "t", "id", "g")
vrai <- ifelse(rs$post == 1, c(`3` = 1.0, `6` = 0.2)[as.character(rs$group)] * (rs$t - rs$group + 1), 0)
cat(sprintf("\nSIMULATION : effet moyen vrai %.3f, Callaway-Sant'Anna %.3f (se %.3f), DiD classique %.3f (se %.3f)\n",
            mean(vrai[rs$post == 1]), aggte(rs, "simple")$overall_att, aggte(rs, "simple")$overall_se,
            twfe_did(sim, "y", "id", "t", "d")$coef, twfe_did(sim, "y", "id", "t", "d")$se))

# 2. Discontinuite : avantage electoral du parti sortant, Senat americain (Lee, 2008)
cat("\nDISCONTINUITE DE REGRESSION, SENAT AMERICAIN\n")
senat <- read.csv("data/rdsenate.csv")
senat <- senat[!is.na(senat$vote), ]
r <- rd_robust(senat$vote, senat$margin, 0)
print(r)
cat("\nrobustesse a la fenetre (h, b = h) :\n")
for (k in c(0.5, 1, 1.5, 2)) {
  rk <- rd_robust(senat$vote, senat$margin, 0, h = k * r$h, b = k * r$h)
  cat(sprintf("  h = %5.2f : effet corrige %.3f, intervalle robuste [%.3f, %.3f]\n", rk$h, rk$tau_bc, rk$ci_rb[1], rk$ci_rb[2]))
}
cat("\nplacebos (seuils fictifs) :\n")
print(round(rd_placebo(senat$vote, senat$margin, 0, placebos = c(-40, -30, -20, -10, 10, 20, 30, 40)), 3))
m <- rd_manipulation(senat$margin, 0, 2)
cat(sprintf("\nmanipulation : %d observations a gauche et %d a droite dans [-2, 2], p = %.3f\n", m$gauche, m$droite, m$pvalue))

# 3. Controle synthetique : cout economique du terrorisme au Pays basque (Abadie et Gardeazabal)
cat("\nCONTROLE SYNTHETIQUE, PAYS BASQUE\n")
basque <- read.csv("data/basque.csv")
speciaux <- c(list(list(variable = "gdpcap", periods = 1960:1969)),
              lapply(c("sec.agriculture", "sec.energy", "sec.industry", "sec.construction",
                       "sec.services.venta", "sec.services.nonventa"), function(v) list(variable = v, periods = seq(1961, 1969, 2))),
              list(list(variable = "popdens", periods = 1969)))
dp <- synth_prep(basque, "regionno", "year", "gdpcap", 17, c(2:16, 18), traitement = 1970, time_optimize = 1960:1969,
                 predictors = c("school.illit", "school.prim", "school.med", "school.high", "school.post.high", "invest"),
                 predictor_periods = 1964:1969, special = speciaux, noms = "regionname")
cat("  optimisation du controle synthetique...\n")
fit <- synth(dp)
print(fit)
e <- synth_ecart(dp, fit$W)
cat(sprintf("ecart moyen de PIB par habitant apres 1970 : %.3f (%.1f %% du niveau du pays basque synthetique)\n",
            mean(e$ecart[e$annee >= 1970]), 100 * mean(e$ecart[e$annee >= 1970] / e$synthetique[e$annee >= 1970])))
cat("  calcul des 17 placebos (environ 8 secondes)...\n")
pl <- synth_placebo(fit)
cat(sprintf("placebos dans l'espace : rapport RMSPE du Pays basque %.1f, p-value %.3f (%d unites)\n",
            pl$rapport_traite, pl$pvalue, nrow(pl$table)))

# Figure
dir.create("figures", showWarnings = FALSE)
png("figures/resultats.png", width = 1500, height = 480, res = 130)
par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))
plot_event_study(dyn, main = "DiD : etude d'evenement (mpdta)")
plot_rd(senat$vote, senat$margin, 0, main = "RDD : Senat americain")
plot_synth(fit, main = "Controle synthetique : Pays basque")
dev.off()
cat("\nfigure ecrite dans figures/resultats.png\n")
