import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import models as M
from data import load_fred_md, make_synthetic
from evaluate import rmse, diebold_mariano


def build_models(target_col, use_torch):
    models = {
        "AR (BIC)": (M.AR(target_col), 1),
        "Diffusion index": (M.DiffusionIndex(), 1),
        "Elastic net": (M.ElasticNetModel(), 6),
    }
    if use_torch and M.HAS_TORCH:
        models["LSTM"] = (M.SequenceModel("lstm"), 12)
        models["Transformer"] = (M.SequenceModel("transformer"), 12)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="fichier FRED-MD (current.csv)")
    ap.add_argument("--target", default="INDPRO")
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no_torch", action="store_true")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    if args.synthetic or args.csv is None:
        df = make_synthetic()
        target = "TARGET"
        start = df.index[int(0.6 * len(df))]
    else:
        df = load_fred_md(args.csv)
        target = args.target
        start = pd.Timestamp(args.start)

    X = df.values
    h = args.h
    T = len(X)
    col = list(df.columns).index(target)
    y_next = np.full(T, np.nan)
    y_next[: T - h] = X[h:, col]

    origins = [t for t in range(T - h) if df.index[t] >= start]
    models = build_models(col, not args.no_torch)
    preds = {name: [] for name in models}
    fitted = {}

    for i, t in enumerate(origins):
        X_hist = X[: t + 1]
        y_hist = y_next[: t - h + 1]
        for name, (model, every) in models.items():
            if name not in fitted or i % every == 0:
                fitted[name] = model.fit(X_hist, y_hist)
            preds[name].append(model.predict(X_hist))
        if i % 50 == 0:
            print(f"origine {i}/{len(origins)}")

    actual = y_next[origins]
    errors = {n: actual - np.array(p) for n, p in preds.items()}

    base = rmse(errors["AR (BIC)"])
    rows = []
    for name, e in errors.items():
        row = {"modele": name, "RMSE": rmse(e), "RMSE relatif a l'AR": rmse(e) / base}
        if name != "AR (BIC)":
            row["stat DM vs AR"], row["p-value"] = diebold_mariano(e, errors["AR (BIC)"], h)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("modele")
    table = table.round({"RMSE": 5, "RMSE relatif a l'AR": 3, "stat DM vs AR": 3, "p-value": 3})
    print(table)
    table.to_csv("results/nowcast_comparaison.csv")

    plt.figure(figsize=(8, 4))
    for name, e in errors.items():
        if name == "AR (BIC)":
            continue
        diff = np.cumsum(errors["AR (BIC)"] ** 2 - e ** 2)
        plt.plot(df.index[origins], diff, label=name)
    plt.axhline(0, color="k", linewidth=0.7)
    plt.ylabel("gain cumule d'erreur quadratique vs AR")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/gain_cumule.png", dpi=130)


if __name__ == "__main__":
    main()
