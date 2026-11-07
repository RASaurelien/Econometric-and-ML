import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import conformal as C
from data import simulate_series, load_series
from evaluate import interval_metrics, rolling_coverage, worst_window_gap
from forecasters import make_lags, ARForecaster, GBMForecaster, QuantileGBM


def run_once(df, args, seed):
    y_all, regime_all = df["y"].values, df["regime"].values
    p = args.lags
    X, y = make_lags(y_all, p)
    regime = regime_all[p:]
    N = len(y)
    n_tr, n_cal = int(args.train_share * N), int(args.cal_share * N)
    tr, cal, te = slice(0, n_tr), slice(n_tr, n_tr + n_cal), slice(n_tr + n_cal, N)

    base = (ARForecaster() if args.base == "ar" else GBMForecaster(seed)).fit(X[tr], y[tr])
    quant = QuantileGBM(args.alpha, seed).fit(X[tr], y[tr])

    pred_cal, pred_te = base.predict(X[cal]), base.predict(X[te])
    qlo_cal, qhi_cal = quant.predict(X[cal])
    qlo_te, qhi_te = quant.predict(X[te])
    y_cal, y_te = y[cal], y[te]
    a = args.alpha

    abs_cal = C.nonconformity(y_cal, pred_cal, pred_cal)
    cqr_cal = C.nonconformity(y_cal, qlo_cal, qhi_cal)

    out = {}
    out["Gaussien"] = C.gaussian_interval(pred_te, y_cal - pred_cal, a)
    out["Conforme fixe"] = C.split_conformal(pred_te, pred_te, abs_cal, a)
    out["Conforme glissant"] = C.online_conformal(y_te, pred_te, pred_te, abs_cal, a, args.window, 0.0)[:2]
    out["ACI"] = C.online_conformal(y_te, pred_te, pred_te, abs_cal, a, args.window, args.gamma)[:2]
    out["CQR fixe"] = C.split_conformal(qlo_te, qhi_te, cqr_cal, a)
    out["CQR + ACI"] = C.online_conformal(y_te, qlo_te, qhi_te, cqr_cal, a, args.window, args.gamma)[:2]

    big = 50 * np.std(y)
    rows = {}
    for name, (lo, hi) in out.items():
        m = interval_metrics(y_te, lo, hi, a, regime[te], big)
        m["pire ecart (fenetre 200)"] = worst_window_gap(y_te, lo, hi, a)
        rows[name] = m
    return pd.DataFrame(rows).T, out, y_te, regime[te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--column", default="y")
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.005)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--lags", type=int, default=5)
    ap.add_argument("--base", choices=["gbm", "ar"], default="gbm")
    ap.add_argument("--train_share", type=float, default=0.35)
    ap.add_argument("--cal_share", type=float, default=0.15)
    ap.add_argument("--no_shift", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    tables, first = [], None
    n_reps = 1 if args.csv else args.reps
    for r in range(n_reps):
        df = load_series(args.csv, args.column) if args.csv else simulate_series(args.T, args.seed + r, not args.no_shift)
        table, out, y_te, reg = run_once(df, args, args.seed + r)
        tables.append(table)
        if first is None:
            first = (out, y_te, reg)
        print(f"repetition {r + 1}/{n_reps}")

    mean = sum(tables) / len(tables)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(f"\nCible de couverture : {1 - args.alpha:.2f} (moyenne sur {len(tables)} series)")
    print(mean.round(3))
    mean.round(4).to_csv("results/comparaison.csv")

    out, y_te, reg = first
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for name, (lo, hi) in out.items():
        ax[0].plot(rolling_coverage(y_te, lo, hi, args.window), label=name, linewidth=1)
    ax[0].axhline(1 - args.alpha, color="k", linestyle="--", linewidth=0.8)
    ax[0].set_xlabel("temps (periode de test)")
    ax[0].set_ylabel(f"couverture glissante ({args.window} obs.)")
    ax[0].legend(fontsize=7)

    last = slice(len(y_te) - 300, len(y_te))
    t = np.arange(300)
    lo, hi = out["CQR + ACI"]
    ax[1].fill_between(t, lo[last], hi[last], alpha=0.3, label="CQR + ACI")
    ax[1].plot(t, y_te[last], color="k", linewidth=0.8, label="serie")
    ax[1].set_xlabel("300 dernieres periodes")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("results/couverture.png", dpi=130)


if __name__ == "__main__":
    main()
