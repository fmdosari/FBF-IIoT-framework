from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, mean, mean
    sem = stats.sem(x)
    half = float(stats.t.ppf((1 + confidence) / 2, len(x) - 1) * sem)
    return mean, mean - half, mean + half


def summarize_final_rounds(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    final = df.sort_values("round").groupby(group_cols + ["seed"], as_index=False).tail(1)
    rows = []
    for keys, g in final.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric in ["auc", "f1", "accuracy", "precision", "recall", "consensus_latency_ms", "energy_kwh"]:
            if metric not in g:
                continue
            m, lo, hi = mean_ci(g[metric])
            row[f"{metric}_mean"] = m
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(df: pd.DataFrame, method_a: str, method_b: str, metric: str = "auc") -> dict:
    final = df.sort_values("round").groupby(["dataset", "aggregator", "attack_fraction", "seed"], as_index=False).tail(1)
    a = final[final["aggregator"] == method_a].set_index(["dataset", "attack_fraction", "seed"])[metric]
    b = final[final["aggregator"] == method_b].set_index(["dataset", "attack_fraction", "seed"])[metric]
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(aligned) < 2:
        return {"n": len(aligned), "t": float("nan"), "p": float("nan")}
    t, p = stats.ttest_rel(aligned["a"], aligned["b"])
    return {"n": int(len(aligned)), "t": float(t), "p": float(p)}
