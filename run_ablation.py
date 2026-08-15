#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pandas as pd

from fbf_iiot.config import AggregationConfig, ExperimentConfig, NetworkConfig, PrivacyConfig
from fbf_iiot.experiment import run_experiment


def main():
    out = Path("results/ablation"); out.mkdir(parents=True, exist_ok=True)
    base = ExperimentConfig(dataset="secom", clients=10, rounds=30, attack="sign_flip", attack_fraction=0.20, seed=42)
    variants = [
        ("full", base, PrivacyConfig(mode="adaptive"), AggregationConfig(method="braga")),
        ("no_blockchain", replace(base, consensus="none"), PrivacyConfig(mode="adaptive"), AggregationConfig(method="braga")),
        ("no_privacy", base, PrivacyConfig(mode="none"), AggregationConfig(method="braga")),
        ("fedavg", replace(base, aggregator="fedavg"), PrivacyConfig(mode="adaptive"), AggregationConfig(method="fedavg")),
        ("pbft_only", replace(base, consensus="pbft"), PrivacyConfig(mode="adaptive"), AggregationConfig(method="braga")),
        ("poa_only", replace(base, consensus="poa"), PrivacyConfig(mode="adaptive"), AggregationConfig(method="braga")),
    ]
    frames = []
    for name, exp, priv, agg in variants:
        print(f"Running {name}")
        hist, _, _ = run_experiment(exp, priv, agg, NetworkConfig(), out_dir=out / name)
        hist["variant"] = name
        frames.append(hist)
    pd.concat(frames, ignore_index=True).to_csv(out / "ablation_all.csv", index=False)


if __name__ == "__main__":
    main()
