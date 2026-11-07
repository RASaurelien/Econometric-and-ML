# Genere les valeurs de reference avec les packages R d'origine (did, rdrobust, Synth).
# A lancer depuis la racine du projet apres data/telecharger.R : Rscript tests/references/generer.R
library(did)
library(rdrobust)
library(Synth)
sortie <- file.path("tests", "references")

# Differences de differences
mpdta <- read.csv("data/mpdta.csv")
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
  cfg <- configs[[nom]]
  r <- att_gt(yname = "lemp", tname = "year", idname = "countyreal", gname = "first.treat",
              xformla = cfg$f, data = mpdta, control_group = cfg$cg, est_method = cfg$m,
              base_period = cfg$base, bstrap = FALSE, cband = FALSE)
  write.csv(data.frame(g = r$group, t = r$t, att = r$att, se = r$se),
            file.path(sortie, paste0("did_", nom, "_attgt.csv")), row.names = FALSE)
  lignes <- list()
  for (ty in c("simple", "dynamic", "group", "calendar")) {
    a <- aggte(r, type = ty, bstrap = FALSE, cband = FALSE)
    lignes[[length(lignes) + 1]] <- data.frame(type = ty, global = 1, indice = NA,
                                               att = a$overall.att, se = a$overall.se)
    if (ty != "simple") {
      lignes[[length(lignes) + 1]] <- data.frame(type = ty, global = 0, indice = a$egt,
                                                 att = a$att.egt, se = a$se.egt)
    }
  }
  write.csv(do.call(rbind, lignes), file.path(sortie, paste0("did_", nom, "_agg.csv")),
            row.names = FALSE)
}

# Discontinuite (la troisieme ligne est la paire de fenetres optimale)
senat <- read.csv("data/rdsenate.csv")
senat <- senat[!is.na(senat$vote), ]
bw <- rdbwselect(senat$vote, senat$margin, c = 0)$bws
paires <- rbind(c(10, 15), c(15, 15), c(bw[1, 1], bw[1, 3]), c(25, 40))
lignes <- lapply(seq_len(nrow(paires)), function(i) {
  r <- rdrobust(senat$vote, senat$margin, c = 0, h = paires[i, 1], b = paires[i, 2])
  data.frame(h = paires[i, 1], b = paires[i, 2], tau = r$coef[1], tau_bc = r$coef[2],
             se = r$se[1], se_rb = r$se[3], n_gauche = r$N_h[1], n_droite = r$N_h[2])
})
write.csv(do.call(rbind, lignes), file.path(sortie, "rd_senat.csv"), row.names = FALSE)

# Controle synthetique
basque <- read.csv("data/basque.csv")
dp <- dataprep(
  foo = basque,
  predictors = c("school.illit", "school.prim", "school.med", "school.high", "school.post.high", "invest"),
  predictors.op = "mean", time.predictors.prior = 1964:1969,
  special.predictors = list(
    list("gdpcap", 1960:1969, "mean"),
    list("sec.agriculture", seq(1961, 1969, 2), "mean"),
    list("sec.energy", seq(1961, 1969, 2), "mean"),
    list("sec.industry", seq(1961, 1969, 2), "mean"),
    list("sec.construction", seq(1961, 1969, 2), "mean"),
    list("sec.services.venta", seq(1961, 1969, 2), "mean"),
    list("sec.services.nonventa", seq(1961, 1969, 2), "mean"),
    list("popdens", 1969, "mean")),
  dependent = "gdpcap", unit.variable = "regionno", unit.names.variable = "regionname",
  time.variable = "year", treatment.identifier = 17, controls.identifier = c(2:16, 18),
  time.optimize.ssr = 1960:1969, time.plot = 1955:1997)
write.csv(cbind(dp$X1, dp$X0), file.path(sortie, "synth_basque_X.csv"))
k <- nrow(dp$X0)
v <- seq_len(k) / sum(seq_len(k))
w_v <- synth(dp, custom.v = v)$solution.w
w_opt <- synth(dp)$solution.w
write.csv(data.frame(v = v), file.path(sortie, "synth_basque_v.csv"), row.names = FALSE)
write.csv(data.frame(w_v = as.vector(w_v), w_opt = as.vector(w_opt)),
          file.path(sortie, "synth_basque_poids.csv"), row.names = FALSE)
