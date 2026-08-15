from __future__ import annotations

import numpy as np
import torch


def poison_vector(vector: torch.Tensor, attack: str, strength: float, seed: int) -> torch.Tensor:
    attack = attack.lower()
    v = vector.detach().cpu().float().clone()
    if attack == "none":
        return v
    if attack == "sign_flip":
        return -float(strength) * v
    if attack == "scale":
        return float(strength) * v
    gen = torch.Generator().manual_seed(seed)
    if attack == "gaussian":
        scale = max(v.std().item(), 1e-6) * float(strength)
        return v + torch.randn(v.shape, generator=gen) * scale
    if attack == "random_direction":
        rnd = torch.randn(v.shape, generator=gen)
        rnd = rnd / (rnd.norm() + 1e-12)
        return rnd * (v.norm() * float(strength))
    raise ValueError(f"Unknown attack: {attack}")
