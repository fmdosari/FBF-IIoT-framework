from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import time
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


@dataclass
class NetworkProfile:
    base_latency_ms: float = 5.0
    jitter_ms: float = 1.0
    packet_loss: float = 0.0


@dataclass
class ConsensusResult:
    success: bool
    mode: str
    latency_ms: float
    votes: int
    required_votes: int
    digest: str


class Validator:
    def __init__(self, validator_id: int, seed: int = 2026):
        self.validator_id = int(validator_id)
        key_bytes = sha256(f"fbf-validator-{seed}-{validator_id}".encode()).digest()
        self.private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
        self.public_key: Ed25519PublicKey = self.private_key.public_key()
        self.reputation = 1.0
        self.uptime = 1.0

    def sign(self, payload: bytes) -> bytes:
        return self.private_key.sign(payload)

    def verify(self, signature: bytes, payload: bytes) -> bool:
        try:
            self.public_key.verify(signature, payload)
            return True
        except Exception:
            return False


class ConsensusEngine:
    def __init__(self, n_validators: int, mode: str = "hybrid", network: NetworkProfile | None = None, seed: int = 2026):
        if n_validators < 1:
            raise ValueError("n_validators must be positive")
        self.validators = [Validator(i, seed=seed) for i in range(n_validators)]
        self.mode = mode.lower()
        self.network = network or NetworkProfile()
        self.rng = np.random.default_rng(seed)
        self._last_mode = "poa"
        self._last_switch_round = -10

    @property
    def f(self) -> int:
        return (len(self.validators) - 1) // 3

    def _deliver(self) -> bool:
        return bool(self.rng.random() >= self.network.packet_loss)

    def _phase_delay(self, message_count: int) -> None:
        if message_count <= 0:
            return
        samples = self.rng.normal(self.network.base_latency_ms, max(self.network.jitter_ms, 0.0), size=message_count)
        delay = max(float(np.max(samples)), 0.0) / 1000.0
        if delay > 0:
            time.sleep(delay)

    def select_mode(self, round_id: int, security_risk: float, sensitivity: float, network_score: float) -> str:
        if self.mode != "hybrid":
            return self.mode
        requested = "pbft" if (security_risk >= 0.35 or sensitivity >= 0.80 or network_score < 0.55) else "poa"
        if requested != self._last_mode and round_id - self._last_switch_round < 3 and security_risk < 0.60:
            return self._last_mode
        if requested != self._last_mode:
            self._last_switch_round = round_id
            self._last_mode = requested
        return requested

    def commit(self, payload: dict, round_id: int, security_risk: float = 0.0,
               sensitivity: float = 0.5, network_score: float = 1.0) -> ConsensusResult:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = sha256(raw).hexdigest()
        msg = digest.encode()
        mode = self.select_mode(round_id, security_risk, sensitivity, network_score)
        start = time.perf_counter()
        if mode == "poa":
            authority = self.validators[round_id % len(self.validators)]
            signature = authority.sign(msg)
            self._phase_delay(1)
            success = self._deliver() and authority.verify(signature, msg)
            votes, required = int(success), 1
        elif mode == "pbft":
            n = len(self.validators)
            required = 2 * self.f + 1
            proposer = self.validators[round_id % n]
            preprepare = proposer.sign(msg)
            if not proposer.verify(preprepare, msg):
                return ConsensusResult(False, mode, 0.0, 0, required, digest)
            self._phase_delay(n)
            prepare_votes = 0
            for v in self.validators:
                if not self._deliver():
                    continue
                sig = v.sign(b"prepare:" + msg)
                if v.verify(sig, b"prepare:" + msg):
                    prepare_votes += 1
            self._phase_delay(n * max(prepare_votes, 1))
            if prepare_votes < required:
                success, votes = False, prepare_votes
            else:
                commit_votes = 0
                for v in self.validators:
                    if not self._deliver():
                        continue
                    sig = v.sign(b"commit:" + msg)
                    if v.verify(sig, b"commit:" + msg):
                        commit_votes += 1
                self._phase_delay(n * max(commit_votes, 1))
                votes = commit_votes
                success = commit_votes >= required
        else:
            raise ValueError(f"Unknown consensus mode: {mode}")
        latency_ms = (time.perf_counter() - start) * 1000.0
        return ConsensusResult(success, mode, latency_ms, votes, required, digest)
