from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import yaml


@dataclass
class NetworkConfig:
    base_latency_ms: float = 5.0
    jitter_ms: float = 1.0
    packet_loss: float = 0.0


@dataclass
class PrivacyConfig:
    mode: str = "adaptive"  # none, dp, he, adaptive
    clip_norm: float = 1.0
    noise_multiplier: float = 1.1
    delta: float = 1e-5
    sensitivity: float = 0.60
    he_poly_modulus_degree: int = 8192
    he_scale_bits: int = 40
    he_chunk_size: int = 2048


@dataclass
class AggregationConfig:
    method: str = "braga"  # fedavg or braga
    mad_z_threshold: float = 3.5
    cosine_threshold: float = 0.05
    reputation_floor: float = 0.05
    reputation_decay: float = 0.80
    reputation_recovery: float = 0.05
    sketch_dim: int = 128


@dataclass
class ExperimentConfig:
    dataset: str = "secom"
    max_samples: int = 10000
    test_size: float = 0.20
    clients: int = 10
    rounds: int = 50
    local_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    dirichlet_alpha: float = 0.5
    min_client_samples: int = 20
    aggregator: str = "braga"
    consensus: str = "hybrid"  # none, pbft, poa, hybrid
    validators: int = 7
    attack: str = "none"  # none, sign_flip, gaussian, scale, random_direction
    attack_fraction: float = 0.0
    attack_strength: float = 5.0
    seed: int = 42
    device: str = "auto"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataclass_dict(obj) -> dict:
    return asdict(obj)
