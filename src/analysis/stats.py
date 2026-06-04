from __future__ import annotations

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / (pooled ** 0.5 + 1e-12))


def run_stats(csv_path: str, metric: str = "dice") -> dict:
    df = pd.read_csv(csv_path)
    groups = [g[metric].values for _, g in df.groupby("group")]
    f_stat, p_val = stats.f_oneway(*groups)

    tukey = pairwise_tukeyhsd(df[metric], df["group"], alpha=0.05)
    ci = df.groupby("group")[metric].agg(["mean", "std", "count"])
    ci["ci95"] = 1.96 * ci["std"] / (ci["count"] ** 0.5)

    d_effects = {}
    group_keys = sorted(df["group"].unique())
    for i, g1 in enumerate(group_keys):
        for g2 in group_keys[i + 1 :]:
            d_effects[f"{g1}_vs_{g2}"] = cohens_d(
                df[df["group"] == g1][metric],
                df[df["group"] == g2][metric],
            )

    return {
        "anova": {"f": float(f_stat), "p": float(p_val)},
        "tukey": tukey.summary().as_text(),
        "ci95": ci.to_dict(),
        "cohens_d": d_effects,
    }
