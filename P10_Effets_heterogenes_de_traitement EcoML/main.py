import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import simulate
import learners as L
from evaluate import summary, qini_curve, group_effects


def build_models(use_torch):
    models = {
        "S-learner": L.SLearner(),
        "T-learner": L.TLearner(),
        "X-learner": L.XLearner(),
        "DR-learner": L.DRLearner(),
        "Foret causale": L.HonestCausalForest(),
    }
    if use_torch and L.HAS_TORCH:
        models["TARNet"] = L.TARNet()
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_test", type=int, default=3000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--observational", action="store_true")
    ap.add_argument("--no_torch", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, curves, last = [], {}, {}

    for r in range(args.reps):
        train = simulate(args.n_train, rng=rng, randomized=not args.observational)
        test = simulate(args.n_test, rng=rng, randomized=True)
        for name, model in build_models(not args.no_torch).items():
            model.fit(train["X"], train["T"], train["Y"])
            tau_hat = model.predict(test["X"])
            res = summary(test["tau"], tau_hat, test["T"], test["Y"])
            res["modele"] = name
            rows.append(res)
            if r == 0:
                curves[name] = qini_curve(tau_hat, test["T"], test["Y"])
                last[name] = (tau_hat, test["tau"])
        print(f"repetition {r + 1}/{args.reps}")

    table = pd.DataFrame(rows).groupby("modele").mean().round(3).sort_values("PEHE")
    print(table)
    table.to_csv("results/comparaison.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, (frac, gain) in curves.items():
        ax[0].plot(frac, gain, label=name)
    ax[0].plot([0, 1], [0, list(curves.values())[0][1][-1]], "k--", label="aleatoire")
    ax[0].set_xlabel("part de la population ciblee")
    ax[0].set_ylabel("gain cumule (Qini)")
    ax[0].legend(fontsize=7)

    for name in ["T-learner", "DR-learner", "Foret causale"]:
        g = group_effects(*last[name])
        ax[1].plot(g[:, 0], g[:, 1], marker="o", label=name)
    lims = ax[1].get_xlim()
    ax[1].plot(lims, lims, "k--")
    ax[1].set_xlabel("effet predit moyen par quintile")
    ax[1].set_ylabel("effet vrai moyen")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("results/uplift_calibration.png", dpi=130)


if __name__ == "__main__":
    main()
