# Regression logistique avec glm.fit : coefficients, probabilites et variance des coefficients
logit <- function(X, d) {
  fit <- suppressWarnings(stats::glm.fit(X, d, family = stats::binomial()))
  k <- ncol(X)
  list(coef = fit$coefficients, p = fit$fitted.values,
       vcov = chol2inv(fit$qr$qr[seq_len(k), seq_len(k), drop = FALSE]))
}

# Ecart-type d'une moyenne a partir de sa fonction d'influence
se_influence <- function(inf) sqrt(mean(inf^2) / length(inf))
