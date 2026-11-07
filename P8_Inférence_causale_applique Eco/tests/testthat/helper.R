# Utilise le package installe (R CMD INSTALL .), sinon charge directement les fichiers de R/.
if (!requireNamespace("minicausal", quietly = TRUE)) {
  racine_r <- if (dir.exists("../../R")) "../../R" else "R"
  for (f in list.files(racine_r, pattern = "\\.R$", full.names = TRUE)) source(f)
} else {
  library(minicausal)
}

# Racine du projet, que les tests soient lances depuis tests/testthat ou depuis la racine.
racine <- if (file.exists("../../DESCRIPTION")) "../.." else "."
lire <- function(...) read.csv(file.path(racine, ...))

# Fonction interne (non exportee) du package, ou de l'environnement global en mode source.
interne <- function(nom) {
  if (requireNamespace("minicausal", quietly = TRUE)) get(nom, envir = asNamespace("minicausal")) else get(nom)
}

# Les references sont generees par tests/references/generer.R (necessite les packages R d'origine).
ref_existe <- function(nom) file.exists(file.path(racine, "tests", "references", nom))
