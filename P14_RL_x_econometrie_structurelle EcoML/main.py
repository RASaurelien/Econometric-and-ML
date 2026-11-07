import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bus_model import BusModel
from nfxp import fit_nfxp, estimate_transitions, state_counts
from ccp import fit_ccp, fit_npl, bootstrap_ccp
from rl_agent import q_learning, replace_prob_from_q

TRUE = {"theta1": 2.5, "rc": 10.0}


def demand_curve(theta1, rc_grid, beta, trans, scale, n_states):
    rates = []
    for rc in rc_grid:
        m = BusModel(theta1, rc, beta, n_states, trans, scale)
        rates.append(m.stationary_replacement_rate()[0])
    return np.array(rates)


def single_run(args):
    rng = np.random.default_rng(args.seed)
    true = BusModel(TRUE["theta1"], TRUE["rc"], args.beta)
    states, actions = true.simulate(args.buses, args.periods, rng)
    n = true.n
    N, R = state_counts(states, actions, n)
    print(f"{states.shape[0]} bus, {states.shape[1]} periodes, taux de remplacement observe {actions.mean():.4f}")

    trans = estimate_transitions(states, actions, n)
    print("increments de kilometrage estimes :", np.round(trans, 3), " vrais :", true.trans)

    nf = fit_nfxp(states, actions, args.beta, trans, true.scale, n)
    cc = fit_ccp(states, actions, args.beta, trans, true.scale, n)
    cc_se = bootstrap_ccp(states, actions, args.beta, true.scale, n, args.bootstrap, rng)
    npl = fit_npl(states, actions, args.beta, trans, true.scale, n)
    npl_se = bootstrap_ccp(states, actions, args.beta, true.scale, n, args.bootstrap, rng, fit=fit_npl)

    est = BusModel(nf["theta"][0], nf["theta"][1], args.beta, n, trans, true.scale)
    print(f"\nQ-learning dans le modele estime par NFXP ({args.rl_steps} pas)...")
    t0 = time.time()
    Q, visits = q_learning(est, args.rl_steps, seed=args.seed)
    rl_time = time.time() - t0

    p_dp = est.replace_prob()
    p_rl = replace_prob_from_q(Q)
    w = visits.sum(1) / visits.sum()
    rl_gap = np.sqrt(np.sum(w * (p_rl - p_dp) ** 2))

    table = pd.DataFrame({
        "theta1 (vrai 2.5)": [nf["theta"][0], cc["theta"][0], npl["theta"][0]],
        "se theta1": [nf["se"][0], cc_se[0], npl_se[0]],
        "RC (vrai 10.0)": [nf["theta"][1], cc["theta"][1], npl["theta"][1]],
        "se RC": [nf["se"][1], cc_se[1], npl_se[1]],
        "temps (s)": [nf["time"], cc["time"], npl["time"]],
    }, index=["NFXP", "CCP Hotz-Miller (inversion)", "CCP NPL (3 iterations)"]).round(3)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print()
    print(table)
    print("\nse NFXP : asymptotiques (la premiere etape sur les transitions est ignoree).")
    print(f"se des deux estimateurs CCP : bootstrap sur les bus ({args.bootstrap} tirages).")
    print(f"\nQ-learning vs programmation dynamique, meme parametres : ecart pondere sur P(remplacer) = {rl_gap:.4f}"
          f" ({rl_time:.1f} s, contre {nf['time']:.1f} s pour tout le NFXP)")
    table.to_csv("results/estimations.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    seen = N >= 5
    freq = np.where(N > 0, R / np.maximum(N, 1), np.nan)
    ax[0].scatter(np.arange(n)[seen], freq[seen], s=np.minimum(N[seen] / 10, 60), color="grey", alpha=0.6,
                  label="frequence observee")
    ax[0].plot(true.replace_prob(), label="vrai")
    ax[0].plot(est.replace_prob(), "--", label="NFXP")
    ax[0].plot(cc["p1"], ":", label="CCP premier stade (logit polynomial)")
    ax[0].plot(npl["p1"], "-.", label="CCP NPL")
    ax[0].plot(np.where(visits.sum(1) > 200, p_rl, np.nan), label="Q-learning", linewidth=1)
    ax[0].set_xlabel("kilometrage (etat)")
    ax[0].set_ylabel("P(remplacer)")
    ax[0].legend(fontsize=8)

    rc_grid = np.linspace(4, 16, 13)
    ax[1].plot(rc_grid, demand_curve(TRUE["theta1"], rc_grid, args.beta, true.trans, true.scale, n), label="vrai")
    ax[1].plot(rc_grid, demand_curve(nf["theta"][0], rc_grid, args.beta, trans, true.scale, n), "--", label="NFXP")
    ax[1].plot(rc_grid, demand_curve(cc["theta"][0], rc_grid, args.beta, trans, true.scale, n), ":", label="CCP inversion")
    ax[1].plot(rc_grid, demand_curve(npl["theta"][0], rc_grid, args.beta, trans, true.scale, n), "-.", label="CCP NPL")
    ax[1].set_xlabel("cout de remplacement RC")
    ax[1].set_ylabel("taux de remplacement par periode")
    ax[1].set_title("contrefactuel : demande de remplacement", fontsize=10)
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/rust.png", dpi=130)


def monte_carlo(args):
    rng = np.random.default_rng(args.seed)
    true = BusModel(TRUE["theta1"], TRUE["rc"], args.beta)
    n = true.n
    names = ["NFXP", "CCP Hotz-Miller", "CCP NPL"]
    out = {k: [] for k in names}
    times = {k: [] for k in names}
    for r in range(args.mc):
        states, actions = true.simulate(args.buses, args.periods, rng)
        trans = estimate_transitions(states, actions, n)
        nf = fit_nfxp(states, actions, args.beta, trans, true.scale, n)
        cc = fit_ccp(states, actions, args.beta, trans, true.scale, n)
        npl = fit_npl(states, actions, args.beta, trans, true.scale, n)
        for key, res in zip(names, (nf, cc, npl)):
            out[key].append(res["theta"])
            times[key].append(res["time"])
        if (r + 1) % 10 == 0:
            print(f"  {r + 1}/{args.mc}")
    truth = np.array([TRUE["theta1"], TRUE["rc"]])
    rows = []
    for name, vals in out.items():
        v = np.array(vals)
        rows.append({"methode": name,
                     "biais theta1": v[:, 0].mean() - truth[0], "RMSE theta1": np.sqrt(np.mean((v[:, 0] - truth[0]) ** 2)),
                     "biais RC": v[:, 1].mean() - truth[1], "RMSE RC": np.sqrt(np.mean((v[:, 1] - truth[1]) ** 2)),
                     "temps moyen (s)": np.mean(times[name])})
    table = pd.DataFrame(rows).set_index("methode").round(3)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(table)
    table.to_csv("results/monte_carlo.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buses", type=int, default=100)
    ap.add_argument("--periods", type=int, default=120)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--rl_steps", type=int, default=600000)
    ap.add_argument("--bootstrap", type=int, default=50)
    ap.add_argument("--mc", type=int, default=0, help="nombre de repetitions Monte Carlo (0 : un seul run)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if args.mc > 0:
        monte_carlo(args)
    else:
        single_run(args)


if __name__ == "__main__":
    main()
