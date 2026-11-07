import numpy as np
import pandas as pd
from doubleml.datasets import fetch_401K

import dml_lib as dl

# effet de l'eligibilite au 401(k) sur le patrimoine financier net (Poterba, Venti, Wise)
data = fetch_401K(return_type="DataFrame")
Y = data["net_tfa"].to_numpy(dtype=float)
D = data["e401"].to_numpy(dtype=float)
X = data[["age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown"]].to_numpy(dtype=float)

rows = []
for name in ["ols_naive", "ols_full", "post_double_selection"]:
    t, se = getattr(dl, name)(Y, D, X)
    rows.append((name, t, se))

learners = ["lasso", "rf", "gbm"] + (["mlp"] if dl.HAS_TORCH else [])
for l in learners:
    t, se = dl.dml_irm(Y, D, X, l, K=5)
    rows.append((f"DML IRM {l}", t, se))

out = pd.DataFrame(rows, columns=["methode", "effet", "se"]).set_index("methode")
print(out.round(0))
