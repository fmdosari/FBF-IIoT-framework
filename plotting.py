from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_convergence(df: pd.DataFrame, output: str | Path) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, g in df.groupby("aggregator"):
        curve = g.groupby("round", as_index=False)["auc"].mean()
        ax.plot(curve["round"], curve["auc"], label=label)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("AUC")
    ax.set_title("Federated convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_robustness(df: pd.DataFrame, output: str | Path) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    final = df.sort_values("round").groupby(["dataset", "aggregator", "attack_fraction", "seed"], as_index=False).tail(1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, g in final.groupby("aggregator"):
        curve = g.groupby("attack_fraction", as_index=False)["auc"].mean()
        ax.plot(curve["attack_fraction"] * 100.0, curve["auc"], marker="o", label=label)
    ax.set_xlabel("Malicious clients (%)")
    ax.set_ylabel("Final AUC")
    ax.set_title("Robustness to Byzantine model poisoning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
