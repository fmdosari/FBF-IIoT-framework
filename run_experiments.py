#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import pandas as pd
import yaml

from fbf_iiot.config import AggregationConfig, ExperimentConfig, NetworkConfig, PrivacyConfig
from fbf_iiot.experiment import run_experiment
from fbf_iiot.plotting import plot_robustness
from fbf_iiot.statistics import summarize_final_rounds


def parse_args():
    p = argparse.ArgumentParser(description="Run the FBF-IIoT publication experiment suite.")
    p.add_argument("--config", default="configs/publication.yaml")
    p.add_argument("--output", default="results/publication")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = ExperimentConfig(**cfg["experiment"])
    privacy = PrivacyConfig(**cfg["privacy"])
    aggregation = AggregationConfig(**cfg["aggregation"])
    network = NetworkConfig(**cfg["network"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset in cfg["suite"]["datasets"]:
        for aggregator in cfg["suite"]["aggregators"]:
            for attack_fraction in cfg["suite"]["attack_fractions"]:
                for seed in cfg["suite"]["seeds"]:
                    attack = "none" if attack_fraction == 0 else cfg["suite"]["attack"]
                    exp = replace(base, dataset=dataset, aggregator=aggregator, attack=attack,
                                  attack_fraction=float(attack_fraction), seed=int(seed))
                    agg = replace(aggregation, method=aggregator)
                    run_name = f"{dataset}_{aggregator}_attack{attack_fraction:.2f}_seed{seed}"
                    print(f"\n=== {run_name} ===")
                    history, _, _ = run_experiment(exp, privacy, agg, network, out_dir=out / "runs" / run_name)
                    frames.append(history)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "all_round_metrics.csv", index=False)
    summary = summarize_final_rounds(all_df, ["dataset", "aggregator", "attack_fraction"])
    summary.to_csv(out / "summary_ci95.csv", index=False)
    plot_robustness(all_df, out / "robustness.png")
    print(f"\nResults written to {out.resolve()}")


if __name__ == "__main__":
    main()
