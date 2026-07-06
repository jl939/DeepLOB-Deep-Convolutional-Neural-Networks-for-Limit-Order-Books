"""FI-2010 limit-order-book dataset loading and windowing.

Ported from the original DeepLOB notebook (run_train_pytorch.ipynb): the first
40 rows are the LOB features, the last 5 rows are labels for 5 horizons.
"""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


def _prepare_x(data: np.ndarray) -> np.ndarray:
    return np.asarray(data[:40, :].T)            # (events, 40)


def _get_label(data: np.ndarray) -> np.ndarray:
    return np.asarray(data[-5:, :].T)            # (events, 5)


def _windowed(X: np.ndarray, Y: np.ndarray, T: int):
    """Sliding window of length T -> (N-T+1, T, 40) and aligned labels."""
    N, D = X.shape
    dataY = Y[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D), dtype=np.float32)
    for i in range(T, N + 1):
        dataX[i - T] = X[i - T:i, :]
    return dataX, dataY


class FI2010Dataset(Dataset):
    """Windowed FI-2010 samples for one prediction horizon `k` (0..4)."""

    def __init__(self, raw: np.ndarray, k: int, T: int):
        x = _prepare_x(raw)
        y = _get_label(raw)
        x, y = _windowed(x, y, T)
        y = y[:, k] - 1                          # labels {1,2,3} -> {0,1,2}
        self.x = torch.from_numpy(x).unsqueeze(1).float()   # (N,1,T,40)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _load_txt(path: str) -> np.ndarray:
    """Load a FI-2010 .txt, caching the parsed array as a sibling .npy so the
    slow np.loadtxt runs only once."""
    npy = path + ".npy"
    if os.path.isfile(npy):
        return np.load(npy)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Extract data/data.zip into the data_dir "
            f"(e.g. `unzip -n data/data.zip -d jupyter_pytorch`).")
    arr = np.loadtxt(path)
    np.save(npy, arr)          # first run pays the parse cost; later runs are fast
    return arr


def build_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train/val/test DataLoaders from the FI-2010 DecPre files."""
    d = cfg.data_dir
    train_raw = _load_txt(os.path.join(d, "Train_Dst_NoAuction_DecPre_CF_7.txt"))
    split = int(np.floor(train_raw.shape[1] * 0.8))
    dec_train, dec_val = train_raw[:, :split], train_raw[:, split:]
    dec_test = np.hstack([
        _load_txt(os.path.join(d, f"Test_Dst_NoAuction_DecPre_CF_{i}.txt"))
        for i in (7, 8, 9)
    ])

    mk = lambda raw: FI2010Dataset(raw, cfg.horizon_k, cfg.window_T)
    train_ds, val_ds, test_ds = mk(dec_train), mk(dec_val), mk(dec_test)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False),
    )


def build_smoke_loaders(cfg, n=512):
    """Tiny synthetic loaders shaped like FI-2010, for fast pipeline checks."""
    g = torch.Generator().manual_seed(cfg.seed)
    def ds(m):
        x = torch.randn(m, 1, cfg.window_T, 40, generator=g)
        y = torch.randint(0, cfg.n_classes, (m,), generator=g)
        return TensorDataset(x, y)
    return (
        DataLoader(ds(n), batch_size=cfg.batch_size, shuffle=True),
        DataLoader(ds(n // 4), batch_size=cfg.batch_size),
        DataLoader(ds(n // 4), batch_size=cfg.batch_size),
    )
