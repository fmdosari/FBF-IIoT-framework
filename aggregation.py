from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class UpdateMetadata:
    client_id: int
    n_samples: int
    norm: float
    sketch: np.ndarray


@dataclass
class BragaDecision:
    accepted: list[int]
    rejected: list[int]
    weights: np.ndarray
    norm_z: np.ndarray
    cosine: np.ndarray


class ReputationBook:
    def __init__(self, floor: float = 0.05, decay: float = 0.80, recovery: float = 0.05):
        self.floor = float(floor)
        self.decay = float(decay)
        self.recovery = float(recovery)
        self.values: dict[int, float] = {}

    def get(self, client_id: int) -> float:
        return self.values.get(client_id, 1.0)

    def update(self, accepted: list[int], rejected: list[int]) -> None:
        for cid in accepted:
            r = self.get(cid)
            self.values[cid] = min(1.0, r + self.recovery * (1.0 - r))
        for cid in rejected:
            self.values[cid] = max(self.floor, self.get(cid) * self.decay)


def count_sketch(vector: torch.Tensor, dim: int, seed: int) -> np.ndarray:
    v = vector.detach().cpu().numpy().astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    bucket = rng.integers(0, dim, size=len(v))
    sign = rng.choice(np.array([-1.0, 1.0]), size=len(v))
    sketch = np.bincount(bucket, weights=sign * v, minlength=dim).astype(np.float64)
    return sketch


def make_metadata(client_id: int, n_samples: int, vector: torch.Tensor, sketch_dim: int, seed: int) -> UpdateMetadata:
    return UpdateMetadata(
        client_id=client_id,
        n_samples=int(n_samples),
        norm=float(torch.linalg.vector_norm(vector).item()),
        sketch=count_sketch(vector, sketch_dim, seed),
    )


def fedavg_weights(metadata: list[UpdateMetadata]) -> np.ndarray:
    sizes = np.asarray([m.n_samples for m in metadata], dtype=float)
    return sizes / sizes.sum()


def braga_gate(metadata: list[UpdateMetadata], reputations: ReputationBook,
               mad_z_threshold: float = 3.5, cosine_threshold: float = 0.05) -> BragaDecision:
    if not metadata:
        raise ValueError("No client updates supplied")
    norms = np.asarray([m.norm for m in metadata], dtype=float)
    med = float(np.median(norms))
    mad = float(np.median(np.abs(norms - med)))
    scale = 1.4826 * mad + 1e-12
    norm_z = np.abs(norms - med) / scale

    sketches = np.stack([m.sketch for m in metadata])
    reference = np.median(sketches, axis=0)
    ref_norm = np.linalg.norm(reference) + 1e-12
    cosine = np.asarray([
        float(np.dot(s, reference) / ((np.linalg.norm(s) + 1e-12) * ref_norm))
        for s in sketches
    ])

    mask = (norm_z <= mad_z_threshold) & (cosine >= cosine_threshold)
    if mask.sum() == 0:
        # Safety fallback: accept the single update closest to the robust norm center.
        mask[int(np.argmin(norm_z))] = True

    accepted = [m.client_id for m, ok in zip(metadata, mask) if ok]
    rejected = [m.client_id for m, ok in zip(metadata, mask) if not ok]
    base = np.asarray([m.n_samples * reputations.get(m.client_id) for m in metadata], dtype=float)
    agreement = np.clip((cosine + 1.0) / 2.0, 0.0, 1.0)
    base *= agreement
    base[~mask] = 0.0
    if base.sum() <= 0:
        base[mask] = 1.0
    weights = base / base.sum()
    reputations.update(accepted, rejected)
    return BragaDecision(accepted, rejected, weights, norm_z, cosine)


def weighted_sum(vectors: list[torch.Tensor], weights: np.ndarray) -> torch.Tensor:
    if len(vectors) != len(weights):
        raise ValueError("Number of vectors and weights must match")
    out = torch.zeros_like(vectors[0], dtype=torch.float32)
    for v, w in zip(vectors, weights):
        out.add_(v.detach().cpu().float(), alpha=float(w))
    return out
