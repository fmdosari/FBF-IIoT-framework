from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .aggregation import ReputationBook, braga_gate, fedavg_weights, make_metadata, weighted_sum
from .attacks import poison_vector
from .config import AggregationConfig, ExperimentConfig, NetworkConfig, PrivacyConfig
from .consensus import ConsensusEngine, NetworkProfile
from .data import load_dataset
from .energy import EnergyMeter
from .federated import apply_delta, choose_device, dirichlet_partition, evaluate, local_train, state_delta, state_to_vector, vector_to_state
from .ledger import AuditLedger
from .model import IndustrialMLP
from .privacy import AdaptivePrivacyController, CKKSBackend, GaussianDP


def _hash_vector(v: torch.Tensor) -> str:
    return sha256(v.detach().cpu().numpy().tobytes()).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(name: str, cache_dir: str, max_samples: int, test_size: float, seed: int):
    bundle = load_dataset(name, cache_dir=cache_dir, max_samples=max_samples, seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(
        bundle.X, bundle.y, test_size=test_size, random_state=seed, stratify=bundle.y
    )
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(Xtr)
    Xte = imp.transform(Xte)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
    return bundle, Xtr, Xte, ytr.astype(np.int64), yte.astype(np.int64)


def run_experiment(exp: ExperimentConfig, privacy: PrivacyConfig | None = None,
                   aggregation: AggregationConfig | None = None, network: NetworkConfig | None = None,
                   cache_dir: str = "data/cache", out_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    privacy = privacy or PrivacyConfig()
    aggregation = aggregation or AggregationConfig(method=exp.aggregator)
    network = network or NetworkConfig()
    _seed_everything(exp.seed)
    device = choose_device(exp.device)
    bundle, Xtr, Xte, ytr, yte = prepare_data(exp.dataset, cache_dir, exp.max_samples, exp.test_size, exp.seed)
    client_splits = dirichlet_partition(ytr, exp.clients, exp.dirichlet_alpha, exp.min_client_samples, exp.seed)
    rng = np.random.default_rng(exp.seed)
    n_mal = int(round(exp.attack_fraction * exp.clients))
    malicious = set(rng.choice(np.arange(exp.clients), size=n_mal, replace=False).tolist()) if n_mal else set()

    global_model = IndustrialMLP(Xtr.shape[1])
    global_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in global_model.state_dict().items())
    reputations = ReputationBook(
        aggregation.reputation_floor, aggregation.reputation_decay, aggregation.reputation_recovery
    )
    dp_clients = {cid: GaussianDP(privacy.clip_norm, privacy.noise_multiplier, privacy.delta) for cid in range(exp.clients)}
    controller = AdaptivePrivacyController()
    he_backend = None
    consensus = None
    ledger = AuditLedger()
    if exp.consensus != "none":
        consensus = ConsensusEngine(
            exp.validators, exp.consensus,
            NetworkProfile(network.base_latency_ms, network.jitter_ms, network.packet_loss),
            seed=exp.seed,
        )

    records = []
    client_records = []
    previous_reject_fraction = 0.0
    for rnd in range(exp.rounds):
        global_model.load_state_dict(global_state)
        observed_latency = max(0.0, float(rng.normal(network.base_latency_ms, max(network.jitter_ms, 0.0))))
        dynamic_sensitivity = min(1.0, privacy.sensitivity + 0.35 * exp.attack_fraction + 0.25 * previous_reject_fraction)
        priv_decision = controller.choose(
            privacy.mode, dynamic_sensitivity,
            observed_latency, network.jitter_ms, network.packet_loss,
        )
        vectors, metadata, losses = [], [], []
        with EnergyMeter() as meter:
            for cid, idx in enumerate(client_splits):
                global_model.load_state_dict(global_state)
                local_state, loss = local_train(
                    global_model, Xtr[idx], ytr[idx], exp.local_epochs, exp.batch_size,
                    exp.learning_rate, exp.weight_decay, device, exp.seed + rnd * 1000 + cid,
                )
                delta_state = state_delta(local_state, global_state)
                vector = state_to_vector(delta_state)
                if cid in malicious:
                    vector = poison_vector(vector, exp.attack, exp.attack_strength, exp.seed + rnd * 2000 + cid)
                if priv_decision.mode == "dp":
                    vector = dp_clients[cid].privatize(vector, exp.seed + rnd * 3000 + cid, sample_rate=1.0)
                vectors.append(vector)
                metadata.append(make_metadata(cid, len(idx), vector, aggregation.sketch_dim, exp.seed + rnd))
                losses.append(loss)

            if aggregation.method == "braga":
                decision = braga_gate(metadata, reputations, aggregation.mad_z_threshold, aggregation.cosine_threshold)
                weights = decision.weights
                rejected = set(decision.rejected)
            elif aggregation.method == "fedavg":
                weights = fedavg_weights(metadata)
                rejected = set()
                decision = None
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation.method}")

            if priv_decision.mode == "he":
                if he_backend is None:
                    he_backend = CKKSBackend(
                        privacy.he_poly_modulus_degree, privacy.he_scale_bits, privacy.he_chunk_size
                    )
                aggregate_vector = he_backend.weighted_sum(vectors, weights)
            else:
                aggregate_vector = weighted_sum(vectors, weights)

            aggregate_delta = vector_to_state(aggregate_vector, global_state)
            global_state = apply_delta(global_state, aggregate_delta)
            global_model.load_state_dict(global_state)

            reject_fraction = len(rejected) / exp.clients
            previous_reject_fraction = reject_fraction
            consensus_mode = "none"
            consensus_latency = 0.0
            consensus_success = True
            if consensus is not None:
                payload = {
                    "dataset": bundle.name,
                    "round": rnd,
                    "privacy": priv_decision.mode,
                    "aggregator": aggregation.method,
                    "aggregate_hash": _hash_vector(aggregate_vector),
                    "rejected_clients": sorted(rejected),
                }
                security_risk = min(1.0, 0.60 * exp.attack_fraction + 0.40 * reject_fraction)
                cres = consensus.commit(
                    payload, rnd, security_risk=security_risk,
                    sensitivity=dynamic_sensitivity, network_score=priv_decision.network_score,
                )
                consensus_mode = cres.mode
                consensus_latency = cres.latency_ms
                consensus_success = cres.success
                if cres.success:
                    ledger.append(payload, cres.mode, cres.latency_ms)
        energy = meter.reading

        metrics = evaluate(global_model, Xte, yte, device)
        eps = max((d.epsilon() for d in dp_clients.values()), default=0.0)
        records.append({
            "dataset": bundle.name,
            "source": bundle.source,
            "seed": exp.seed,
            "round": rnd + 1,
            "clients": exp.clients,
            "malicious_clients": len(malicious),
            "attack": exp.attack,
            "attack_fraction": exp.attack_fraction,
            "aggregator": aggregation.method,
            "privacy_mode": priv_decision.mode,
            "privacy_sensitivity": priv_decision.sensitivity,
            "network_score": priv_decision.network_score,
            "epsilon": eps,
            "delta": privacy.delta,
            "consensus_mode": consensus_mode,
            "consensus_success": consensus_success,
            "consensus_latency_ms": consensus_latency,
            "rejected_updates": len(rejected),
            "mean_local_loss": float(np.mean(losses)),
            "wall_seconds": energy.wall_seconds,
            "cpu_joules": energy.cpu_joules,
            "gpu_joules": energy.gpu_joules,
            "energy_kwh": energy.total_kwh,
            **metrics,
        })
        for i, m in enumerate(metadata):
            client_records.append({
                "dataset": bundle.name,
                "seed": exp.seed,
                "round": rnd + 1,
                "client_id": m.client_id,
                "malicious": m.client_id in malicious,
                "n_samples": m.n_samples,
                "update_norm": m.norm,
                "reputation": reputations.get(m.client_id),
                "accepted": m.client_id not in rejected,
                "weight": float(weights[i]),
            })

    history = pd.DataFrame(records)
    clients = pd.DataFrame(client_records)
    summary = {
        "experiment": asdict(exp),
        "privacy": asdict(privacy),
        "aggregation": asdict(aggregation),
        "network": asdict(network),
        "dataset_source": bundle.source,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "n_features": int(Xtr.shape[1]),
        "model_parameters": int(global_model.trainable_parameters),
        "malicious_client_ids": sorted(malicious),
        "ledger_blocks": len(ledger.blocks),
        "ledger_valid": ledger.verify(),
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        history.to_csv(out / "round_metrics.csv", index=False)
        clients.to_csv(out / "client_metrics.csv", index=False)
        pd.DataFrame(ledger.to_records()).to_csv(out / "ledger.csv", index=False)
        (out / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return history, clients, summary
