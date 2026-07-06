#!/usr/bin/env python
"""Compress only the CONV layers of DeepLOB (keep the fast built-in nn.LSTM),
then retrain on the FULL dataset for many epochs. Because the LSTM stays as
cuDNN, training is fast enough for a proper full-data retrain -- the setting
that gives the best chance of matching or *beating* the baseline (MPO can act
as a regularizer).

Example:
  python scripts/compress_deeplob_conv.py --bond 8 --epochs 30 --device mps
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import compress_deeplob_lstm as C
import __main__
__main__.deeplob = C.deeplob

from deeplob_mpo.mpo import MPOConv2d
from deeplob_mpo.compress import _get_parent
from deeplob_mpo.config import Config
from deeplob_mpo.data import build_loaders
from deeplob_mpo.engine import evaluate, fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jupyter_pytorch/best_val_model_pytorch")
    p.add_argument("--bond", type=int, default=8)
    p.add_argument("--n-cores", type=int, default=3)
    p.add_argument("--min-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-subset", type=int, default=10000,
                   help="cap the per-epoch validation check for speed")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev, D = args.device, args.bond

    cfg = Config(device=dev, batch_size=args.batch_size)
    train_loader, val_loader, test_loader = build_loaders(cfg)   # FULL data
    val_mon = C.cap(val_loader, args.val_subset, args.batch_size)   # fast per-epoch check

    model = torch.load(args.ckpt, map_location=dev, weights_only=False).to(dev)
    _, base_acc = evaluate(model, test_loader, dev)              # full-test baseline
    base_params = C.count(model)
    print(f"baseline: {base_params:,} params, full-test acc {base_acc:.4f}\n", flush=True)

    # compress every eligible Conv2d; keep nn.LSTM intact (fast cuDNN)
    convs = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Conv2d)
             and min(m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]) >= args.min_dim]
    print(f"{'layer':<10}{'dense':>8}{'mpo':>8}{'ratio':>8}", flush=True)
    print("-" * 34, flush=True)
    for name, conv in convs:
        dense = sum(p.numel() for p in conv.parameters())
        new, rel = MPOConv2d.from_conv2d(conv, D, args.n_cores, warm_start=True)
        if new.n_params() >= dense:
            print(f"{name:<10}{dense:>8}{new.n_params():>8}{'skip':>8}", flush=True)
            continue
        parent, key = _get_parent(model, name)
        setattr(parent, key, new.to(dev))
        print(f"{name:<10}{dense:>8}{new.n_params():>8}{new.n_params()/dense:>8.3f}", flush=True)

    after = C.count(model)
    print("-" * 34, flush=True)
    print(f"{'MODEL':<10}{base_params:>8}{after:>8}{after/base_params:>8.3f}", flush=True)

    _, acc_pre = evaluate(model, test_loader, dev)
    print(f"\nafter MPO (no retrain)  full-test acc {acc_pre:.4f}", flush=True)

    fit(model, train_loader, val_mon, epochs=args.epochs, lr=args.lr,
        weight_decay=1e-5, device=dev, ckpt_path="checkpoints/deeplob_conv_mpo.pt")
    model.load_state_dict(torch.load("checkpoints/deeplob_conv_mpo.pt", map_location=dev))
    _, acc_full = evaluate(model, test_loader, dev)
    print(f"\nafter retrain           full-test acc {acc_full:.4f}", flush=True)
    print(f"whole model: {base_params:,} -> {after:,} ({base_params/after:.2f}x smaller), "
          f"acc {base_acc:.4f} -> {acc_full:.4f}  "
          f"({'BEATS' if acc_full > base_acc else 'below'} baseline)", flush=True)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
