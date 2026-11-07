# A lancer une fois depuis la racine du projet (necessite internet) : Rscript data/telecharger.R
paquets <- c("did", "rdrobust", "Synth")
manquants <- setdiff(paquets, rownames(installed.packages()))
if (length(manquants) > 0) {
  install.packages(manquants, repos = "https://cloud.r-project.org")
}

dir.create("data", showWarnings = FALSE)
data(mpdta, package = "did")
write.csv(mpdta, "data/mpdta.csv", row.names = FALSE)
data(rdrobust_RDsenate, package = "rdrobust")
write.csv(rdrobust_RDsenate, "data/rdsenate.csv", row.names = FALSE)
data(basque, package = "Synth")
write.csv(basque, "data/basque.csv", row.names = FALSE)
