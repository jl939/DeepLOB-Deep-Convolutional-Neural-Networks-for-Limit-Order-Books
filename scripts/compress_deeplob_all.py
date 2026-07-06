#!/usr/bin/env python
"""MPO-compress EVERY eligible weight matrix in DeepLOB (all conv layers + both
LSTM matrices), skipping ones too small to benefit, then fine-tune.

Example:
  python scripts/compress_deeplob_all.py --bond 8 --finetune-epochs 3 --device mps
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
from deeplob_mpo.compress import mpoify_linear, _get_parent
from deeplob_mpo.config import Config
from deeplob_mpo.data import build_loaders
from deeplob_mpo.engine import evaluate, fit


def matrix_dims(m):
    """(out, in) of the layer viewed as a matrix."""
    if isinstance(m, nn.Conv2d):
        return m.out_channels, m.in_channels * m.kernel_size[0] * m.kernel_size[1]
    return m.out_features, m.in_features       # nn.Linear


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jupyter_pytorch/best_val_model_pytorch")
    p.add_argument("--bond", type=int, default=8)
    p.add_argument("--n-cores", type=int, default=3)
    p.add_argument("--min-dim", type=int, default=8,
                   help="skip matrices whose smaller side is below this")
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-subset", type=int, default=40000)
    p.add_argument("--eval-subset", type=int, default=20000)
    p.add_argument("--device", default=None)
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

    model.lstm = LinearLSTM.from_nn_lstm(model.lstm).to(dev)   # expose LSTM matrices

    # collect every Conv2d / Linear, decide eligibility
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))
               and min(matrix_dims(m)) >= args.min_dim]

    print(f"{'layer':<14}{'dense':>8}{'mpo':>8}{'ratio':>8}{'TT-SVD':>9}", flush=True)
    print("-" * 47, flush=True)
    n_done = 0
    for name, mod in targets:
        dense = sum(p.numel() for p in mod.parameters())
        if isinstance(mod, nn.Conv2d):
            new, rel = MPOConv2d.from_conv2d(mod, D, args.n_cores, warm_start=True)
        else:
            new, rel = mpoify_linear(mod, D, args.n_cores, warm_start=True)
        if new.n_params() >= dense:                # MPO bigger than dense -> skip
            print(f"{name:<14}{dense:>8}{new.n_params():>8}{'skip':>8}{'':>9}", flush=True)
            continue
        parent, key = _get_parent(model, name)
        setattr(parent, key, new.to(dev))
        n_done += 1
        print(f"{name:<14}{dense:>8}{new.n_params():>8}{new.n_params()/dense:>8.3f}{rel:>9.3f}",
              flush=True)

    after = C.count(model)
    print("-" * 47, flush=True)
    print(f"{'MODEL':<14}{base_params:>8}{after:>8}{after/base_params:>8.3f}  "
          f"({n_done} layers compressed)", flush=True)

    _, acc_pre = evaluate(model, eval_test, dev)
    print(f"\nafter MPO (no fine-tune)  test acc ~{acc_pre:.4f}", flush=True)

    if args.finetune_epochs > 0:
        ft_train = C.cap(train_loader, args.train_subset, args.batch_size, shuffle=True)
        fit(model, ft_train, ft_val, epochs=args.finetune_epochs, lr=args.finetune_lr,
            weight_decay=1e-5, device=dev, ckpt_path="checkpoints/deeplob_all_mpo.pt")
        model.load_state_dict(torch.load("checkpoints/deeplob_all_mpo.pt", map_location=dev))
        _, acc_full = evaluate(model, test_loader, dev)
        print(f"after fine-tune           full test acc {acc_full:.4f}", flush=True)
        print(f"\nwhole model: {base_params:,} -> {after:,} params "
              f"({base_params/after:.2f}x smaller), acc {base_acc:.3f} -> {acc_full:.3f}",
              flush=True)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
