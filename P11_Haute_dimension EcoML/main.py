import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz

import hd_lib as hd

SELECTORS = ["lasso_cv", "lasso_plugin", "elastic_net", "ridge"]


def selection_experiment(n, p, s, rhos, reps, rng):
    rows = []
    for rho in rhos:
        for r in range(reps):
            _, _, X, beta = hd.simulate(n, p, s, rho, 0.0, rng)
            Xte = rng.normal(size=(2000, p))
            Xte = Xte @ np.linalg.cholesky(
                toeplitz(rho ** np.arange(p))).T
            yte = Xte @ beta
            y = X @ beta + rng.normal(size=n)
            for name in SELECTORS:
                m = hd.fit_selector(name, X, y)
                prec, rec = hd.support_scores(m.coef_, beta)
                rows.append({
                    "rho": rho, "methode": name, "precision": prec, "recall": rec,
                    "erreur l2 beta": np.linalg.norm(m.coef_ - beta),
                    "RMSE prevision": np.sqrt(np.mean((m.predict(Xte) - yte) ** 2)),
                })
    return pd.DataFrame(rows).groupby(["rho", "methode"]).mean().round(3)


def inference_experiment(n, p, s, rhos, reps, alpha, rng):
    methods = {
        "OLS naif": hd.naive_ols,
        "post-lasso naif (CV)": hd.naive_post_lasso,
        "post-double-selection (plug-in)": hd.post_double_selection,
        "post-double-selection (CV)": lambda Y, D, X: hd.post_double_selection(Y, D, X, "lasso_cv"),
    }
    rows = []
    for rho in rhos:
        est = {k: [] for k in methods}
        se = {k: [] for k in methods}
        for r in range(reps):
            Y, D, X, _ = hd.simulate(n, p, s, rho, alpha, rng)
            for name, fn in methods.items():
                a, b = fn(Y, D, X)
                est[name].append(a)
                se[name].append(b)
        for name in methods:
            e, b = np.array(est[name]), np.array(se[name])
            rows.append({
                "rho": rho, "methode": name,
                "biais": e.mean() - alpha, "RMSE": np.sqrt(np.mean((e - alpha) ** 2)),
                "couverture": np.mean(np.abs(e - alpha) <= 1.96 * b),
            })
    return pd.DataFrame(rows).set_index(["rho", "methode"]).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--s", type=int, default=10)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rhos = [0.0, 0.5, 0.9]

    sel = selection_experiment(args.n, args.p, args.s, rhos, args.reps, rng)
    print("Selection de variables et prevision (n=%d, p=%d, s=%d)" % (args.n, args.p, args.s))
    print(sel)
    sel.to_csv("results/selection.csv")

    inf = inference_experiment(args.n, args.p, args.s, rhos, args.reps, args.alpha, rng)
    print("\nInference sur l'effet de D")
    print(inf)
    inf.to_csv("results/inference.csv")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for name in SELECTORS:
        d = sel.xs(name, level="methode")
        ax[0].plot(d.index, d["precision"], marker="o", label=name)
        ax[1].plot(d.index, d["recall"], marker="o", label=name)
    ax[0].set_title("precision")
    ax[1].set_title("recall")
    for a in ax:
        a.set_xlabel("correlation entre regresseurs (rho)")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("results/support_vs_correlation.png", dpi=130)


if __name__ == "__main__":
    main()
