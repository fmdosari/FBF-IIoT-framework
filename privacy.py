from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass
class PrivacyDecision:
    mode: str
    network_score: float
    sensitivity: float


class AdaptivePrivacyController:
    """Selects DP or HE from task sensitivity and measured network quality."""

    def __init__(self, he_sensitivity_threshold: float = 0.72, he_network_threshold: float = 0.60):
        self.he_sensitivity_threshold = float(he_sensitivity_threshold)
        self.he_network_threshold = float(he_network_threshold)

    @staticmethod
    def network_score(latency_ms: float, jitter_ms: float, packet_loss: float) -> float:
        latency_term = math.exp(-max(latency_ms, 0.0) / 200.0)
        jitter_term = math.exp(-max(jitter_ms, 0.0) / 50.0)
        loss_term = max(0.0, 1.0 - min(max(packet_loss, 0.0), 1.0))
        return float(0.45 * latency_term + 0.20 * jitter_term + 0.35 * loss_term)

    def choose(self, requested_mode: str, sensitivity: float, latency_ms: float,
               jitter_ms: float, packet_loss: float) -> PrivacyDecision:
        mode = requested_mode.lower()
        score = self.network_score(latency_ms, jitter_ms, packet_loss)
        if mode == "adaptive":
            mode = "he" if sensitivity >= self.he_sensitivity_threshold and score >= self.he_network_threshold else "dp"
        if mode not in {"none", "dp", "he"}:
            raise ValueError(f"Unknown privacy mode: {requested_mode}")
        return PrivacyDecision(mode, score, float(sensitivity))


class GaussianDP:
    def __init__(self, clip_norm: float, noise_multiplier: float, delta: float):
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.steps = 0
        self.sample_rate = 1.0
        try:
            from opacus.accountants import RDPAccountant
            self.accountant = RDPAccountant()
        except Exception:
            self.accountant = None

    def privatize(self, vector: torch.Tensor, seed: int, sample_rate: float = 1.0) -> torch.Tensor:
        v = vector.detach().cpu().float().clone()
        norm = torch.linalg.vector_norm(v)
        scale = min(1.0, self.clip_norm / (float(norm) + 1e-12))
        v.mul_(scale)
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(v.shape, generator=gen) * (self.noise_multiplier * self.clip_norm)
        v.add_(noise)
        self.steps += 1
        self.sample_rate = float(sample_rate)
        if self.accountant is not None:
            self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sample_rate)
        return v

    def epsilon(self) -> float:
        if self.accountant is not None:
            return float(self.accountant.get_epsilon(delta=self.delta))
        # Conservative single-mechanism composition fallback. This is labeled as an upper bound.
        if self.steps == 0:
            return 0.0
        per_step = math.sqrt(2.0 * math.log(1.25 / self.delta)) / max(self.noise_multiplier, 1e-12)
        return float(self.steps * per_step)


class CKKSBackend:
    """Chunked CKKS weighted aggregation using TenSEAL."""

    def __init__(self, poly_modulus_degree: int = 8192, scale_bits: int = 40, chunk_size: int = 2048):
        try:
            import tenseal as ts
        except ImportError as exc:
            raise RuntimeError("HE mode requires TenSEAL. Install with: pip install tenseal") from exc
        self.ts = ts
        self.chunk_size = int(chunk_size)
        self.secret_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=int(poly_modulus_degree),
            coeff_mod_bit_sizes=[60, scale_bits, scale_bits, 60],
        )
        self.secret_context.global_scale = 2 ** int(scale_bits)
        self.public_context = self.secret_context.copy()
        self.public_context.make_context_public()

    def weighted_sum(self, vectors: list[torch.Tensor], weights: np.ndarray) -> torch.Tensor:
        if len(vectors) != len(weights):
            raise ValueError("Number of vectors and weights must match")
        arrays = [v.detach().cpu().numpy().astype(np.float64, copy=False) for v in vectors]
        n = len(arrays[0])
        if any(len(a) != n for a in arrays):
            raise ValueError("All HE vectors must have the same length")
        pieces = []
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            enc_sum = None
            for arr, w in zip(arrays, weights):
                enc = self.ts.ckks_vector(self.public_context, arr[start:end].tolist()) * float(w)
                enc_sum = enc if enc_sum is None else enc_sum + enc
            dec = np.asarray(enc_sum.decrypt(secret_key=self.secret_context.secret_key()), dtype=np.float32)
            pieces.append(dec[: end - start])
        return torch.from_numpy(np.concatenate(pieces))
