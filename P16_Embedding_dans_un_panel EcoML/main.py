import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fe_models as FE
from dgp import simulate_panel, split_cells
from embed_model import HAS_TORCH

SCENARIOS = {
    "additif": dict(scenario="additive", n_units=300, n_times=10, obs_prob=1.0),
    "interactif": dict(scenario="interactive", n_units=300, n_times=10, obs_prob=1.0),
    "peu d'obs. par unite": dict(scenario="additive", n_units=1000, n_times=20, obs_prob=0.15),
}
BETA = 1.0


def fit_models(train, n_units, n_times, use_torch, seed):
    out = {}
    out["MCO empile"] = FE.pooled_ols(train)
    out["Effets fixes (TWFE)"] = FE.twfe(train, n_units, n_times)
    out["Effets fixes interactifs (Bai, r=2)"] = FE.interactive_fe(train, n_units, n_times, rank=2)
    b, pred, _ = FE.ridge_effects(train, n_units, n_times, mundlak=False, seed=seed)
    out["Effets aleatoires (ridge)"] = (b, pred)
    b, pred, _ = FE.ridge_effects(train, n_units, n_times, mundlak=True, seed=seed)
    out["Effets aleatoires + Mundlak"] = (b, pred)
    if use_torch:
        from embed_model import fit_embedding
        mundlak = ("x", "xbar_u", "xbar_t")
        configs = (("Embeddings additifs", ("x",), False),
                   ("Embeddings additifs + Mundlak", mundlak, False),
                   ("Embeddings interactifs (d=4) + Mundlak", mundlak, True))
        for name, cov, interactive in configs:
            beta, predict, _ = fit_embedding(train, n_units, n_times, covariates=cov,
                                             interactive=interactive, seed=seed)
            out[name] = (beta, predict)
    return out


def run_scenario(params, reps, use_torch, seed):
    rows = []
    for r in range(reps):
        df = simulate_panel(seed=seed + r, beta=BETA, **params)
        train, test = split_cells(df, 0.2, seed=seed + r)
        train, test = FE.add_group_means(train, test, params["n_units"], params["n_times"])
        n_units, n_times = params["n_units"], params["n_times"]
        for name, (beta, predict) in fit_models(train, n_units, n_times, use_torch, seed + r).items():
            err = test["y"].values - predict(test)
            rows.append({"modele": name, "beta": beta, "rmse_test": np.sqrt(np.mean(err ** 2))})
    return pd.DataFrame(rows)


def single_table(args):
    df = simulate_panel("additive", n_units=300, n_times=10, seed=args.seed)
    rows = [("MCO empile", *FE.pooled_cluster_se(df)), ("Effets fixes (TWFE)", *FE.twfe_cluster_se(df))]
    table = pd.DataFrame(rows, columns=["modele", "beta", "se (cluster par unite)"]).set_index("modele").round(3)
    print(f"Un jeu de donnees, scenario additif (vrai beta = {BETA})")
    print(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no_torch", action="store_true")
    ap.add_argument("--single", action="store_true", help="un seul jeu de donnees avec ecarts-types clusterises")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    if args.single:
        single_table(args)
        return

    use_torch = HAS_TORCH and not args.no_torch
    if not use_torch:
        print("torch indisponible ou desactive : les modeles a embeddings sont ignores\n")

    summary, draws = [], {}
    for label, params in SCENARIOS.items():
        res = run_scenario(params, args.reps, use_torch, args.seed)
        draws[label] = res
        g = res.groupby("modele", sort=False)
        tab = pd.DataFrame({
            "biais beta": g["beta"].mean() - BETA,
            "RMSE beta": g["beta"].apply(lambda b: np.sqrt(np.mean((b - BETA) ** 2))),
            "RMSE test": g["rmse_test"].mean(),
        }).round(3)
        tab.insert(0, "scenario", label)
        summary.append(tab)
        print(f"=== {label} (bruit irreductible : RMSE test = 1) ===")
        print(tab.drop(columns="scenario"))
        print()
    pd.concat(summary).to_csv("results/comparaison.csv")

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(5 * len(SCENARIOS), 4), sharey=True)
    for ax, (label, res) in zip(axes, draws.items()):
        names = list(res["modele"].unique())
        ax.boxplot([res.loc[res["modele"] == n, "beta"] for n in names], tick_labels=[n[:14] for n in names])
        ax.axhline(BETA, color="red", linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
    axes[0].set_ylabel("beta estime")
    plt.tight_layout()
    plt.savefig("results/beta_estimes.png", dpi=130)


if __name__ == "__main__":
    main()
