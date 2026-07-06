#!/usr/bin/env python
"""Compress DeepLOB's heaviest layers together: the LSTM input matrix + the
three biggest conv layers, all as MPO, then fine-tune. This is the honest
whole-model test -- with multiple layers compressed at once there is little
free capacity left to 'route around' the bottleneck.

Example:
  python scripts/compress_deeplob_full.py --bond 8 --finetune-epochs 3 --device mps
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

from deeplob_mpo.models import LinearLSTM
from deeplob_mpo.mpo import MPOConv2d
from deeplob_mpo.compress import mpoify_linear
from deeplob_mpo.config import Config
from deeplob_mpo.data import build_loaders
from deeplob_mpo.engine import evaluate, fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jupyter_pytorch/best_val_model_pytorch")
    p.add_argument("--bond", type=int, default=8)
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-subset", type=int, default=40000)
    p.add_argument("--eval-subset", type=int, default=20000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev, D = args.device, args.bond

    cfg = Config(device=dev, batch_size=args.batch_size)
    train_loader, val_loader, test_loader = build_loaders(cfg)
    ft_val = C.cap(val_loader, args.eval_subset, args.batch_size)
    eval_test = C.cap(test_loader, args.eval_subset, args.batch_size)

    model = torch.load(args.ckpt, map_location=dev, weights_only=False).to(dev)
    _, base_acc = evaluate(model, eval_test, dev)
    base_params = C.count(model)
    print(f"baseline: {base_params:,} params, test acc ~{base_acc:.4f}\n", flush=True)

    # LSTM input matrix
    model.lstm = LinearLSTM.from_nn_lstm(model.lstm).to(dev)
    print(f"{'layer':<12}{'dense':>9}{'mpo':>8}{'ratio':>8}{'TT-SVD err':>12}", flush=True)
    print("-" * 49, flush=True)

    def report(name, dense, mpo_p, rel):
        print(f"{name:<12}{dense:>9}{mpo_p:>8}{mpo_p/dense:>8.3f}{rel:>12.3f}", flush=True)

    ih = model.lstm.ih
    dense = ih.weight.numel() + ih.bias.numel()
    mpo, rel = mpoify_linear(ih, D, n_cores=3, warm_start=True)
    model.lstm.ih = mpo.to(dev)
    report("lstm.ih", dense, mpo.n_params(), rel)

    # three biggest conv layers: (module, index, name)
    conv_targets = [(model.conv3, 0, "conv3.0"),
                    (model.inp1, 3, "inp1.3"),
                    (model.inp2, 3, "inp2.3")]
    for seq, idx, name in conv_targets:
        conv = seq[idx]
        dense = conv.weight.numel() + (conv.bias.numel() if conv.bias is not None else 0)
        mconv, rel = MPOConv2d.from_conv2d(conv, D, n_cores=3, warm_start=True)
        seq[idx] = mconv.to(dev)
        report(name, dense, mconv.n_params(), rel)

    after = C.count(model)
    print("-" * 49, flush=True)
    print(f"{'MODEL':<12}{base_params:>9}{after:>8}{after/base_params:>8.3f}", flush=True)

    _, acc_pre = evaluate(model, eval_test, dev)
    print(f"\nafter MPO (no fine-tune)  test acc ~{acc_pre:.4f}", flush=True)

    if args.finetune_epochs > 0:
        ft_train = C.cap(train_loader, args.train_subset, args.batch_size, shuffle=True)
        fit(model, ft_train, ft_val, epochs=args.finetune_epochs, lr=args.finetune_lr,
            weight_decay=1e-5, device=dev, ckpt_path="checkpoints/deeplob_full_mpo.pt")
        model.load_state_dict(torch.load("checkpoints/deeplob_full_mpo.pt", map_location=dev))
        _, acc_sub = evaluate(model, eval_test, dev)
        _, acc_full = evaluate(model, test_loader, dev)
        print(f"after fine-tune           test acc ~{acc_sub:.4f}  (full test {acc_full:.4f})", flush=True)
        print(f"\nwhole model: {base_params:,} -> {after:,} params "
              f"({base_params/after:.2f}x smaller), acc {base_acc:.3f} -> {acc_full:.3f}", flush=True)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
