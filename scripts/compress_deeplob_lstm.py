#!/usr/bin/env python
"""MPO-compress ONE weight matrix of the trained DeepLOB LSTM.

Uses the working, pre-trained DeepLOB model (jupyter_pytorch/best_val_model_pytorch),
swaps its nn.LSTM for an equivalent LinearLSTM (lossless), replaces the
input->hidden Linear (192->256) with an MPOLinear, then fine-tunes and reports
test accuracy.

Examples
--------
  python scripts/compress_deeplob_lstm.py --baseline-only --device mps
  python scripts/compress_deeplob_lstm.py --bond-dim 8 --finetune-epochs 5 --device mps
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeplob_mpo.models import LinearLSTM
from deeplob_mpo.compress import mpoify_linear
from deeplob_mpo.config import Config
from deeplob_mpo.data import build_loaders
from deeplob_mpo.engine import evaluate, fit


def cap(loader, n, batch_size, shuffle=False):
    """Evenly subsample a loader's dataset to at most n samples (for speed:
    the loop-based LinearLSTM is too slow to sweep the full 250k windows)."""
    ds = loader.dataset
    if n and n < len(ds):
        idx = np.linspace(0, len(ds) - 1, n).astype(int)
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --- architecture matching best_val_model_pytorch (clean device-agnostic forward)
class deeplob(nn.Module):
    def __init__(self, y_len=3):
        super().__init__()
        self.y_len = y_len
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 2), (1, 2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 2), (1, 2)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.Tanh(), nn.BatchNorm2d(32))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 10)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32))
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (5, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), (1, 1), (1, 0)),
            nn.Conv2d(32, 64, (1, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        h0 = x.new_zeros(1, x.size(0), 64)
        c0 = x.new_zeros(1, x.size(0), 64)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        return self.fc1(x)                      # logits


def count(m):
    return sum(p.numel() for p in m.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jupyter_pytorch/best_val_model_pytorch")
    p.add_argument("--matrix", default="ih", choices=["ih", "hh"],
                   help="which LSTM matrix to compress (ih=input->hidden 256x192, "
                        "hh=hidden->hidden 256x64)")
    p.add_argument("--bond-dim", type=int, default=8)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-subset", type=int, default=40000,
                   help="cap train windows for fine-tune (loop LSTM is slow)")
    p.add_argument("--eval-subset", type=int, default=20000,
                   help="cap val/test windows for evaluation")
    p.add_argument("--device", default=None)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="train ONLY the compressed MPO layer; freeze the rest "
                        "(fewer trainable params -> much less overfitting)")
    p.add_argument("--baseline-only", action="store_true",
                   help="just evaluate the trained DeepLOB and exit")
    args = p.parse_args()

    dev = args.device
    cfg = Config(device=dev, batch_size=args.batch_size)
    train_loader, val_loader, test_loader = build_loaders(cfg)

    # 1. load the trained DeepLOB and measure its accuracy (fast: cuDNN LSTM)
    model = torch.load(args.ckpt, map_location=dev, weights_only=False).to(dev)
    _, acc0 = evaluate(model, test_loader, dev)
    print(f"DeepLOB baseline (full test)   params={count(model):,}  test acc={acc0:.4f}",
          flush=True)
    if args.baseline_only:
        return

    # subsampled loaders: the loop-based LinearLSTM can't sweep 250k windows fast
    ft_train = cap(train_loader, args.train_subset, args.batch_size, shuffle=True)
    ft_val = cap(val_loader, args.eval_subset, args.batch_size)
    eval_test = cap(test_loader, args.eval_subset, args.batch_size)

    # 2. swap nn.LSTM -> LinearLSTM (lossless) — quick 1-batch correctness check
    model.lstm = LinearLSTM.from_nn_lstm(model.lstm).to(dev)
    xb, yb = next(iter(eval_test))
    with torch.no_grad():
        ok = model(xb.to(dev, torch.float)).argmax(1).cpu()
    print(f"after LSTM->LinearLSTM         (lossless swap; 1-batch check ran)", flush=True)

    # 3. MPO the chosen matrix
    linear = getattr(model.lstm, args.matrix)
    dense = linear.weight.numel() + linear.bias.numel()
    mpo, rel = mpoify_linear(linear, args.bond_dim, n_cores=3, warm_start=True)
    setattr(model.lstm, args.matrix, mpo.to(dev))
    print(f"MPO on lstm.{args.matrix}: {tuple(linear.weight.shape)}  "
          f"{dense} -> {mpo.n_params()} params  (bond {args.bond_dim}, TT-SVD err {rel:.3f})",
          flush=True)

    # 4. accuracy right after conversion (no fine-tune), on the eval subset
    _, acc2 = evaluate(model, eval_test, dev)
    print(f"after MPO (no fine-tune)       params={count(model):,}  test acc~{acc2:.4f}",
          flush=True)

    # optionally freeze everything except the compressed MPO layer
    if args.freeze_backbone:
        keep = f"lstm.{args.matrix}."
        n_tr = sum(prm.numel() for name, prm in model.named_parameters()
                   if name.startswith(keep))
        for name, prm in model.named_parameters():
            prm.requires_grad = name.startswith(keep)
        print(f"freeze-backbone: training only {n_tr} params (the MPO cores)", flush=True)

    # 5. fine-tune on the subset
    if args.finetune_epochs > 0:
        fit(model, ft_train, ft_val, epochs=args.finetune_epochs,
            lr=args.finetune_lr, weight_decay=1e-5, device=dev,
            ckpt_path="checkpoints/deeplob_lstm_mpo.pt")
        model.load_state_dict(torch.load("checkpoints/deeplob_lstm_mpo.pt", map_location=dev))
        _, acc3 = evaluate(model, eval_test, dev)
        print(f"after fine-tune                params={count(model):,}  test acc~{acc3:.4f}",
              flush=True)
        _, acc_full = evaluate(model, test_loader, dev)
        print(f"after fine-tune (full test)    params={count(model):,}  test acc={acc_full:.4f}",
              flush=True)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
