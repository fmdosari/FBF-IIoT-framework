from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import time


@dataclass
class Block:
    index: int
    timestamp: float
    previous_hash: str
    payload_hash: str
    consensus_mode: str
    consensus_latency_ms: float
    block_hash: str


class AuditLedger:
    def __init__(self):
        self.blocks: list[Block] = []

    @staticmethod
    def payload_hash(payload: dict) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(raw).hexdigest()

    def append(self, payload: dict, consensus_mode: str, consensus_latency_ms: float) -> Block:
        index = len(self.blocks)
        previous = self.blocks[-1].block_hash if self.blocks else "0" * 64
        ph = self.payload_hash(payload)
        ts = time.time()
        raw = f"{index}|{ts:.9f}|{previous}|{ph}|{consensus_mode}|{consensus_latency_ms:.9f}".encode()
        bh = sha256(raw).hexdigest()
        block = Block(index, ts, previous, ph, consensus_mode, consensus_latency_ms, bh)
        self.blocks.append(block)
        return block

    def verify(self) -> bool:
        for i, b in enumerate(self.blocks):
            expected_prev = self.blocks[i-1].block_hash if i else "0" * 64
            if b.previous_hash != expected_prev:
                return False
            raw = f"{b.index}|{b.timestamp:.9f}|{b.previous_hash}|{b.payload_hash}|{b.consensus_mode}|{b.consensus_latency_ms:.9f}".encode()
            if sha256(raw).hexdigest() != b.block_hash:
                return False
        return True

    def to_records(self) -> list[dict]:
        return [asdict(b) for b in self.blocks]
