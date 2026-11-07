import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold

import dml_lib as dl
from simulate import dgp_plr, dgp_irm

# nom : (modele, design, nombre de covariables)
SCENARIOS = {
    "plr_smooth": ("plr", "smooth", 50),
    "plr_complex": ("plr", "complex", 300),
    "irm_complex": ("irm", "complex", 50),
    "plr_linear": ("plr", "linear", 50),
    "irm_linear": ("irm", "linear", 50),
}
DEFAULT = ["plr_smooth", "plr_complex", "irm_complex"]


def draw_data(sc, n, theta, rng):
    model, design, p = SCENARIOS[sc]
    if model == "plr":
        return dgp_plr(n, p, theta, rng, design)
    return dgp_irm(n, p, theta, rng, design)


def get_methods(model, learners, n_rep):
    methods = {"OLS naif": dl.ols_naive, "OLS tous les X": dl.ols_full}
    if model == "plr":
        methods["Post-lasso"] = dl.post_lasso
    methods["Post-double-selection"] = dl.post_double_selection
    for name in learners:
        if model == "plr":
            methods[f"DML {name}"] = lambda Y, D, X, l=name: dl.dml_plr(Y, D, X, l, n_rep=n_rep)
        else:
            methods[f"DML {name}"] = lambda Y, D, X, l=name: dl.dml_irm(Y, D, X, l)
    return methods


def one_rep(sc, rep, n, theta, learners, n_rep, seed):
    rng = np.random.default_rng([seed, rep])
    Y, D, X = draw_data(sc, n, theta, rng)
    methods = get_methods(SCENARIOS[sc][0], learners, n_rep)
    return {name: fn(Y, D, X) for name, fn in methods.items()}


def nuisance_report(Y, D, X, learners, model):
    # R2 hors echantillon des fonctions de nuisance, pour expliquer le classement
    rows = []
    kf = KFold(5, shuffle=True, random_state=0)
    for name in learners:
        pred_y = cross_val_predict(dl.make_learner(name), X, Y, cv=kf)
        r2_y = 1 - np.sum((Y - pred_y) ** 2) / np.sum((Y - Y.mean()) ** 2)
        if model == "plr":
            pred_d = cross_val_predict(dl.make_learner(name), X, D, cv=kf)
        else:
            skf = StratifiedKFold(5, shuffle=True, random_state=0)
            pred_d = cross_val_predict(dl.make_learner(name, "clf"), X, D, cv=skf,
                                       method="predict_proba")[:, 1]
        r2_d = 1 - np.sum((D - pred_d) ** 2) / np.sum((D - D.mean()) ** 2)
        rows.append({"learner": name, "R2 Y|X": r2_y, "R2 D|X": r2_d})
    return pd.DataFrame(rows).set_index("learner").round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT, choices=list(SCENARIOS))
    ap.add_argument("--learners", nargs="+", default=["lasso", "rf", "gbm", "mlp"])
    ap.add_argument("--n_rep", type=int, default=1, help="repetitions du cross-fitting (PLR)")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    learners = [l for l in args.learners if l != "mlp" or dl.HAS_TORCH]
    if len(learners) < len(args.learners):
        print("torch introuvable, le learner mlp est ignore")

    os.makedirs("results", exist_ok=True)

    for sc in args.scenarios:
        model, design, p = SCENARIOS[sc]
        print(f"\n=== {sc} (n={args.n}, p={p}) ===")
        Y, D, X = draw_data(sc, args.n, args.theta, np.random.default_rng(args.seed))
        print(nuisance_report(Y, D, X, learners, model))

        out = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(one_rep)(sc, r, args.n, args.theta, learners, args.n_rep, args.seed)
            for r in range(args.reps))

        names = list(out[0])
        est = {k: np.array([o[k][0] for o in out]) for k in names}
        err = {k: np.array([o[k][1] for o in out]) for k in names}

        rows = []
        for name in names:
            e, s = est[name], err[name]
            covered = (np.abs(e - args.theta) <= 1.96 * s)
            rows.append({
                "methode": name,
                "biais": np.nanmean(e) - args.theta,
                "ecart_type": np.nanstd(e),
                "RMSE": np.sqrt(np.nanmean((e - args.theta) ** 2)),
                "se moyen": np.nanmean(s),
                "couverture": np.nanmean(covered),
            })
        table = pd.DataFrame(rows).set_index("methode").round(3)
        print(table)
        table.to_csv(f"results/{sc}.csv")

        fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 3),
                                 sharex=True, sharey=True)
        for ax, name in zip(axes, names):
            ax.hist(est[name][~np.isnan(est[name])], bins=20, color="steelblue")
            ax.axvline(args.theta, color="red", linestyle="--")
            ax.set_title(name, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"results/{sc}.png", dpi=130)
        plt.close()


if __name__ == "__main__":
    main()
