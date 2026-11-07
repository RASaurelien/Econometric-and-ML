import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace

import models as M
import realized as R
from evaluate import qlike, squared_error, diebold_mariano, mincer_zarnowitz
from simulate_intraday import simulate_returns

HAR = ["d", "w", "m"]
SHAR = ["rsv_neg", "rsv_pos", "w", "m"]
ALL = ["d", "w", "m", "rsv_neg", "rsv_pos"]


def build_models(use_torch):
    # nom : (modele, frequence de reestimation en jours)
    models = {
        "HAR": (M.LinearModel(HAR), 1),
        "SHAR": (M.LinearModel(SHAR), 1),
        "Ridge 22 retards": (M.RidgeLags(), 5),
        "Foret aleatoire": (M.ForestModel(ALL), 20),
    }
    if use_torch and M.HAS_TORCH:
        models["MLP"] = (M.MLPModel(ALL), 40)
        models["LSTM"] = (M.LSTMModel(), 40)
    return models


def print_har_regression(data, h):
    valid = np.flatnonzero(~np.isnan(data.y) & data.X[HAR].notna().all(axis=1).values)
    X = sm.add_constant(data.X[HAR].values[valid])
    fit = sm.OLS(data.y[valid], X).fit(cov_type="HAC", cov_kwds={"maxlags": max(5, h)})
    print("HAR en echantillon complet (erreurs de Newey-West)")
    for name, b, se in zip(["const"] + HAR, fit.params, fit.bse):
        print(f"  {name:6s} {b:8.3f}  (se {se:.3f})")
    print(f"  R2 = {fit.rsquared:.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="prix intraday : colonne dates puis colonne price")
    ap.add_argument("--freq", default="5min")
    ap.add_argument("--days", type=int, default=1000, help="jours simules")
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--no_torch", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    if args.csv:
        meas = R.measures_from_csv(args.csv, args.freq)
    else:
        returns, idx = simulate_returns(args.days, seed=args.seed)
        meas = R.measures_from_array(returns, idx)
    feats, seq = R.build_features(meas)
    y = R.make_target(meas, args.h)
    data = SimpleNamespace(X=feats, seq=seq, y=y)
    h, W = args.h, args.window

    print_har_regression(data, h)

    first = int(np.flatnonzero(feats.notna().all(axis=1).values)[0])
    origins = np.arange(first + W + h, len(meas) - h)
    models = build_models(not args.no_torch)
    har_ref = M.LinearModel(HAR)

    preds = {name: [] for name in models}
    sigma2 = []
    for i, t in enumerate(origins):
        train = np.arange(t - h - W + 1, t - h + 1)
        har_ref.fit(data, train)
        sigma2.append(np.var(data.y[train] - har_ref.predict(data, train)))
        for name, (model, every) in models.items():
            if i % every == 0:
                model.fit(data, train)
            preds[name].append(model.predict(data, np.array([t]))[0])
        if i % 250 == 0:
            print(f"origine {i}/{len(origins)}")

    sigma2 = np.array(sigma2)
    log_real = y[origins]
    rv_real = np.exp(log_real)
    losses = {}
    rows = []
    for name in models:
        f_log = np.array(preds[name])
        f_rv = np.exp(f_log + 0.5 * sigma2)
        ql = qlike(rv_real, f_rv)
        se_log = (log_real - f_log) ** 2
        losses[name] = {"qlike": ql, "mse_log": se_log}
        a, b, r2 = mincer_zarnowitz(log_real, f_log)
        rows.append({"modele": name, "MSE log": se_log.mean(),
                     "MSE niveau (x1e8)": squared_error(rv_real, f_rv).mean() * 1e8,
                     "QLIKE": ql.mean(), "MZ pente": b, "MZ R2": r2})
    table = pd.DataFrame(rows).set_index("modele")
    for name in table.index:
        if name == "HAR":
            continue
        table.loc[name, "DM QLIKE p-value"] = diebold_mariano(losses[name]["qlike"], losses["HAR"]["qlike"], h)[1]
        table.loc[name, "DM MSE log p-value"] = diebold_mariano(losses[name]["mse_log"], losses["HAR"]["mse_log"], h)[1]
    table = table.round(4)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(table)
    table.to_csv("results/comparaison.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    dates = meas.index[origins]
    ax[0].plot(dates[-250:], np.sqrt(rv_real[-250:] * 252) * 100, color="grey", label="vol realisee")
    for name in ["HAR", "SHAR"] + [n for n in models if n in ("MLP", "LSTM")]:
        ax[0].plot(dates[-250:], np.sqrt(np.exp(np.array(preds[name][-250:]) + 0.5 * sigma2[-250:]) * 252) * 100,
                   label=name, linewidth=1)
    ax[0].set_ylabel("volatilite annualisee (%)")
    ax[0].legend(fontsize=7)
    for name in models:
        if name == "HAR":
            continue
        ax[1].plot(dates, np.cumsum(losses["HAR"]["qlike"] - losses[name]["qlike"]), label=name)
    ax[1].axhline(0, color="k", linewidth=0.7)
    ax[1].set_ylabel("gain cumule de QLIKE vs HAR")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("results/volatilite.png", dpi=130)


if __name__ == "__main__":
    main()
