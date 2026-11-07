import numpy as np

# X (n, k), e (n,), bread = (X'X)^-1, c = codes des individus.


# Variance classique : sigma^2 estime avec ddl degres de liberte.
def classique(e, bread, ddl):
    return e @ e / ddl * bread


# Variance clusterisee par individu. corr est le facteur de petit echantillon.
def cluster(X, e, bread, c, N, corr=1.0):
    u = X * e[:, None]
    scores = np.zeros((N, X.shape[1]))
    np.add.at(scores, c, u)
    return corr * bread @ (scores.T @ scores) @ bread
