from collections import OrderedDict
import numpy as np
import torch

from fbf_iiot.aggregation import ReputationBook, braga_gate, make_metadata, weighted_sum
from fbf_iiot.attacks import poison_vector
from fbf_iiot.consensus import ConsensusEngine, NetworkProfile
from fbf_iiot.ledger import AuditLedger


def test_braga_rejects_extreme_update():
    vecs = [torch.ones(100) * 0.1, torch.ones(100) * 0.11, torch.ones(100) * 0.09, torch.ones(100) * -20.0]
    meta = [make_metadata(i, 100, v, 32, 7) for i, v in enumerate(vecs)]
    rep = ReputationBook()
    dec = braga_gate(meta, rep, mad_z_threshold=3.5, cosine_threshold=0.0)
    assert 3 in dec.rejected
    assert np.isclose(dec.weights.sum(), 1.0)


def test_poison_sign_flip():
    v = torch.tensor([1.0, -2.0])
    out = poison_vector(v, "sign_flip", 3.0, 1)
    assert torch.allclose(out, torch.tensor([-3.0, 6.0]))


def test_poa_consensus_and_ledger():
    engine = ConsensusEngine(4, "poa", NetworkProfile(0, 0, 0), seed=1)
    payload = {"round": 1, "hash": "abc"}
    result = engine.commit(payload, 1)
    assert result.success
    ledger = AuditLedger()
    ledger.append(payload, result.mode, result.latency_ms)
    assert ledger.verify()


def test_pbft_threshold():
    engine = ConsensusEngine(4, "pbft", NetworkProfile(0, 0, 0), seed=2)
    result = engine.commit({"round": 1}, 1)
    assert result.success
    assert result.required_votes == 3


def test_weighted_sum():
    out = weighted_sum([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], np.array([0.25, 0.75]))
    assert torch.allclose(out, torch.tensor([2.5, 3.5]))


def test_end_to_end_experiment_without_external_data(monkeypatch):
    from fbf_iiot import experiment as exmod
    from fbf_iiot.config import AggregationConfig, ExperimentConfig, NetworkConfig, PrivacyConfig
    from fbf_iiot.data import DatasetBundle

    rng = np.random.default_rng(4)
    X = rng.normal(size=(160, 8)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    Xtr, Xte, ytr, yte = X[:120], X[120:], y[:120], y[120:]
    bundle = DatasetBundle(X, y, "test-only", "in-memory-test")
    monkeypatch.setattr(exmod, "prepare_data", lambda *a, **k: (bundle, Xtr, Xte, ytr, yte))

    exp = ExperimentConfig(dataset="secom", clients=3, rounds=2, local_epochs=1, batch_size=16,
                           min_client_samples=10, aggregator="braga", consensus="poa", seed=4)
    hist, clients, summary = exmod.run_experiment(
        exp, PrivacyConfig(mode="none"), AggregationConfig(method="braga"),
        NetworkConfig(base_latency_ms=0, jitter_ms=0, packet_loss=0),
    )
    assert len(hist) == 2
    assert summary["ledger_valid"]
    assert len(clients) == 6
