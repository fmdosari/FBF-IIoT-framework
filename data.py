from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import requests
import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    name: str
    source: str


def _subsample(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not max_samples or len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    keep = []
    per_class = max_samples // max(1, len(np.unique(y)))
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        take = min(len(idx), per_class)
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    remaining = max_samples - len(keep)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(len(y)), np.asarray(keep, dtype=int))
        if len(pool):
            keep.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    keep = np.asarray(keep, dtype=int)
    rng.shuffle(keep)
    return X[keep], y[keep]


def load_secom(cache_dir: str | Path = "data/cache", max_samples: int | None = None, seed: int = 42, **_) -> DatasetBundle:
    import io
    import zipfile
    cache = Path(cache_dir).expanduser().resolve() / "secom"
    cache.mkdir(parents=True, exist_ok=True)
    data_file = cache / "secom.data"
    labels_file = cache / "secom_labels.data"
    if not data_file.exists() or not labels_file.exists():
        url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(cache)
    if not data_file.exists() or not labels_file.exists():
        raise FileNotFoundError("SECOM archive was downloaded but secom.data or secom_labels.data was not found.")
    Xdf = pd.read_csv(data_file, sep=r"\s+", header=None).apply(pd.to_numeric, errors="coerce")
    yraw = pd.to_numeric(pd.read_csv(labels_file, sep=r"\s+", header=None).iloc[:, 0], errors="coerce").fillna(-1).to_numpy()
    X = Xdf.to_numpy(dtype=np.float32)
    y = (yraw > 0).astype(np.int64)
    X, y = _subsample(X, y, max_samples, seed)
    return DatasetBundle(X, y, "SECOM", "https://archive.ics.uci.edu/dataset/179/secom")


def _find_file(root: Path, pattern: str) -> Path | None:
    rx = re.compile(pattern, flags=re.I)
    for p in root.rglob("*"):
        if p.is_file() and rx.search(p.name):
            return p
    return None


def load_cmapss(cache_dir: str | Path = "data/cache", max_samples: int | None = 10000, seed: int = 42, **_) -> DatasetBundle:
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    train = _find_file(cache, r"^train_FD001\.txt$")
    if train is None:
        try:
            import kagglehub
            path = Path(kagglehub.dataset_download("behrad3d/nasa-cmaps"))
            train = _find_file(path, r"^train_FD001\.txt$")
        except Exception as exc:
            raise RuntimeError(
                "C-MAPSS FD001 was not found. Install kagglehub and allow it to download "
                "the public NASA C-MAPSS mirror 'behrad3d/nasa-cmaps'."
            ) from exc
    if train is None:
        raise FileNotFoundError("train_FD001.txt was not found after dataset download.")

    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    df = pd.read_csv(train, sep=r"\s+", header=None, names=cols, usecols=range(26))
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = last_cycle - df["cycle"]
    y = (rul <= 30).astype(np.int64).to_numpy()
    features = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    X = df[features].to_numpy(dtype=np.float32)
    X, y = _subsample(X, y, max_samples, seed)
    return DatasetBundle(X, y, "C-MAPSS FD001", "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_tep(cache_dir: str | Path = "data/cache", max_samples: int | None = 10000, seed: int = 42, **_) -> DatasetBundle:
    cache = Path(cache_dir).expanduser().resolve() / "tep"
    base = "https://raw.githubusercontent.com/mv-per/tennessee-eastman-dataset/main/simulations/mode_1"
    files: list[tuple[Path, int]] = []
    normal = _download(f"{base}/mode1_normal_50.xlsx", cache / "mode1_normal_50.xlsx")
    files.append((normal, 0))
    for fault in range(1, 22):
        name = f"mode1_{fault}_1.xlsx"
        p = _download(f"{base}/faults/{name}", cache / name)
        files.append((p, 1))

    frames, labels = [], []
    n_features = None
    for path, label in files:
        df = pd.read_excel(path)
        num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        if num.empty:
            num = df.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(axis=1, how="all")
        if n_features is None:
            n_features = min(52, num.shape[1])
        n_features = min(n_features, num.shape[1])
        frames.append(num)
        labels.append(label)
    X_parts, y_parts = [], []
    for df, label in zip(frames, labels):
        arr = df.iloc[:, :n_features].to_numpy(dtype=np.float32)
        X_parts.append(arr)
        y_parts.append(np.full(len(arr), label, dtype=np.int64))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    X, y = _subsample(X, y, max_samples, seed)
    return DatasetBundle(X, y, "Tennessee Eastman Process", "https://github.com/mv-per/tennessee-eastman-dataset")


def _binary_label(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return int(float(value) != 0.0)
    text = str(value).strip().lower()
    return 0 if text in {"0", "normal", "healthy", "ok", "nominal", "false", "none", "no_anomaly"} else 1


def load_factorynet(cache_dir: str | Path = "data/cache", max_samples: int | None = 10000, seed: int = 42, **_) -> DatasetBundle:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets to load FactoryNet: pip install datasets pyarrow") from exc

    ds = load_dataset("Forgis/FactoryNet", "normalized", split="train", streaming=True, cache_dir=str(cache_dir))
    feature_names = list(ds.features.keys()) if getattr(ds, "features", None) else []
    label_candidates = [c for c in feature_names if "anomaly" in c.lower() and c.lower().startswith("ctx_")]
    if not label_candidates:
        label_candidates = [c for c in feature_names if "label" in c.lower()]
    if not label_candidates:
        raise RuntimeError("FactoryNet schema contains no anomaly label column in the normalized configuration.")
    label_col = label_candidates[0]
    feature_cols = [
        c for c in feature_names
        if c != label_col and c.startswith(("setpoint_", "feedback_", "effort_"))
    ]
    if not feature_cols:
        raise RuntimeError("FactoryNet normalized configuration contains no S-E-F numeric feature columns.")
    feature_cols = feature_cols[:128]

    rows, ys = [], []
    limit = max_samples or 10000
    for row in ds:
        vals = []
        for c in feature_cols:
            v = row.get(c)
            try:
                vals.append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                vals.append(np.nan)
        rows.append(vals)
        ys.append(_binary_label(row.get(label_col)))
        if len(rows) >= limit:
            break
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise RuntimeError("FactoryNet sample did not contain both normal and anomalous labels; increase max_samples.")
    return DatasetBundle(X, y, "FactoryNet", "https://huggingface.co/datasets/Forgis/FactoryNet")


LOADERS = {
    "secom": load_secom,
    "cmapss": load_cmapss,
    "tep": load_tep,
    "factorynet": load_factorynet,
}


def load_dataset(name: str, **kwargs) -> DatasetBundle:
    key = name.strip().lower()
    if key not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(LOADERS)}")
    bundle = LOADERS[key](**kwargs)
    if bundle.X.ndim != 2 or bundle.y.ndim != 1 or len(bundle.X) != len(bundle.y):
        raise RuntimeError(f"Invalid dataset shape returned for {name}: X={bundle.X.shape}, y={bundle.y.shape}")
    if len(np.unique(bundle.y)) < 2:
        raise RuntimeError(f"Dataset {name} has fewer than two classes after loading.")
    return bundle
