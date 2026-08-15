# Reproducibility protocol

## Software environment

Use Python 3.10 or 3.11 and install the full dependencies with:

```bash
pip install -e ".[all]"
```

Record the output of:

```bash
python --version
python -m pip freeze > results/software_versions.txt
```

## Publication experiment

Run:

```bash
python scripts/run_experiments.py --config configs/publication.yaml --output results/publication
```

The configuration fixes the datasets, model training parameters, client count, communication rounds, Dirichlet concentration, aggregation method, attacks, attack fractions, privacy parameters, consensus configuration, and five random seeds.

## Statistical reporting

Report the final round mean and 95 percent confidence interval across independent seeds. When comparing two aggregation methods under matched dataset, attack fraction, and seed settings, use the paired statistical test implementation in `src/fbf_iiot/statistics.py`.

## TPS and latency

Run `scripts/benchmark_consensus.py`. Throughput is calculated as successful commits divided by the measured wall clock benchmark duration. Commit latency is measured with `time.perf_counter()` around the consensus protocol path. Network delay, jitter, and packet loss are explicit command line parameters.

## Energy

The code reads Linux Intel RAPL and NVIDIA NVML energy counters when supported by the machine. If neither counter is available, energy is not estimated and the result remains empty. Do not replace missing hardware energy values with constants.

## Privacy budget

DP is applied to clipped client update vectors. Each client maintains its own accountant across communication rounds. With Opacus installed, epsilon is obtained from the RDP accountant for the configured delta. The reported experiment epsilon is the maximum cumulative epsilon across participating clients. If Opacus is unavailable, the implementation reports a conservative composition upper bound.

## Dataset integrity

The experiment loaders do not create synthetic substitutes. Dataset download or parsing failure stops the run with an explicit error. This prevents accidental publication of synthetic values under a source dataset name.
