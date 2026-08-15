#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import pandas as pd

from fbf_iiot.consensus import ConsensusEngine, NetworkProfile


def main():
    p = argparse.ArgumentParser(description="Benchmark measured PBFT, PoA, and hybrid consensus execution.")
    p.add_argument("--validators", nargs="+", type=int, default=[4, 7, 10, 16])
    p.add_argument("--transactions", type=int, default=100)
    p.add_argument("--latency-ms", type=float, default=2.0)
    p.add_argument("--jitter-ms", type=float, default=0.5)
    p.add_argument("--packet-loss", type=float, default=0.0)
    p.add_argument("--output", default="results/consensus_benchmark.csv")
    args = p.parse_args()
    rows = []
    for n in args.validators:
        for mode in ["pbft", "poa", "hybrid"]:
            engine = ConsensusEngine(n, mode, NetworkProfile(args.latency_ms, args.jitter_ms, args.packet_loss), seed=2026+n)
            successes = 0
            latencies = []
            start = time.perf_counter()
            for r in range(args.transactions):
                result = engine.commit({"transaction": r, "payload": "model_update_commitment"}, r,
                                       security_risk=0.40 if r % 10 == 0 else 0.10,
                                       sensitivity=0.75, network_score=0.90)
                successes += int(result.success)
                if result.success:
                    latencies.append(result.latency_ms)
            elapsed = time.perf_counter() - start
            rows.append({
                "validators": n,
                "mode": mode,
                "submitted": args.transactions,
                "committed": successes,
                "commit_rate": successes / args.transactions,
                "wall_seconds": elapsed,
                "measured_tps": successes / elapsed if elapsed > 0 else float("nan"),
                "mean_latency_ms": sum(latencies) / len(latencies) if latencies else float("nan"),
            })
            print(rows[-1])
    df = pd.DataFrame(rows)
    from pathlib import Path
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
