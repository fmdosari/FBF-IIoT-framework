from __future__ import annotations

from collections import OrderedDict
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, min_size: int, seed: int) -> list[np.ndarray]:
    if n_clients < 1:
        raise ValueError("n_clients must be positive")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(500):
        splits = [[] for _ in range(n_clients)]
        for cls in classes:
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
            chunks = np.split(idx, cuts)
            for i, chunk in enumerate(chunks):
                splits[i].extend(chunk.tolist())
        arrays = [np.asarray(s, dtype=int) for s in splits]
        if min(len(s) for s in arrays) >= min_size:
            for s in arrays:
                rng.shuffle(s)
            return arrays
    raise RuntimeError("Unable to create a Dirichlet partition satisfying min_client_samples; lower the client count or min size.")


def state_to_vector(state: OrderedDict | dict) -> torch.Tensor:
    return torch.cat([v.detach().cpu().float().reshape(-1) for v in state.values()])


def vector_to_state(vector: torch.Tensor, template: OrderedDict | dict) -> OrderedDict:
    out = OrderedDict()
    pos = 0
    for k, v in template.items():
        n = v.numel()
        out[k] = vector[pos:pos+n].reshape(v.shape).to(dtype=v.dtype)
        pos += n
    if pos != vector.numel():
        raise ValueError("Vector length does not match state template")
    return out


def state_delta(local_state: OrderedDict | dict, global_state: OrderedDict | dict) -> OrderedDict:
    return OrderedDict((k, local_state[k].detach().cpu() - global_state[k].detach().cpu()) for k in global_state)


def apply_delta(global_state: OrderedDict | dict, delta: OrderedDict | dict) -> OrderedDict:
    return OrderedDict((k, global_state[k].detach().cpu() + delta[k].detach().cpu()) for k in global_state)


def local_train(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
                lr: float, weight_decay: float, device: torch.device, seed: int) -> tuple[dict, float]:
    torch.manual_seed(seed)
    model = copy.deepcopy(model).to(device)
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, generator=gen)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
    state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
    return state, float(np.mean(losses)) if losses else float("nan")


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> dict:
    model = copy.deepcopy(model).to(device).eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()), batch_size=512)
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.sigmoid(model(xb.to(device))).detach().cpu().numpy()
            probs.append(p)
            labels.append(yb.numpy())
    p = np.concatenate(probs)
    y_true = np.concatenate(labels)
    thresholds = np.linspace(0.10, 0.90, 81)
    f1s = [f1_score(y_true, p >= t, zero_division=0) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(f1s))])
    pred = (p >= threshold).astype(int)
    try:
        auc = float(roc_auc_score(y_true, p))
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "threshold": threshold,
    }
