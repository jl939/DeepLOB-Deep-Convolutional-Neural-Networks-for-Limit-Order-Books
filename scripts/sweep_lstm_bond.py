#!/usr/bin/env python
"""Sweep MPO bond dimension for the DeepLOB LSTM input->hidden matrix.

For each bond in the list: start from the trained DeepLOB, MPO-compress lstm.ih
to that bond, fine-tune on a data subset, and record accuracy before/after.
Prints one table: bond -> parameters -> accuracy.

Example:
  python scripts/sweep_lstm_bond.py --bonds 2 4 6 8 10 --device mps
"""
import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import compress_deeplob_lstm as C          # reuse deeplob class + cap()
import __main__
__main__.deeplob = C.deeplob               # let torch.load unpickle the checkpoint

from deeplob_mpo.models import LinearLSTM
from deeplob_mpo.compress import mpoify_linear
from deeplob_mpo.config import Config
from deeplob_mpo.data import build_loaders
from deeplob_mpo.engine import evaluate, fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="jupyter_pytorch/best_val_model_pytorch")
    p.add_argument("--bonds", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    p.add_argument("--finetune-epochs", type=int, default=3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-subset", type=int, default=40000)
    p.add_argument("--eval-subset", type=int, default=20000)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    dev = args.device
    cfg = Config(device=dev, batch_size=args.batch_size)
    train_loader, val_loader, test_loader = build_loaders(cfg)
    ft_val = C.cap(val_loader, args.eval_subset, args.batch_size)
    eval_test = C.cap(test_loader, args.eval_subset, args.batch_size)

    # baseline (fast cuDNN LSTM)
    base = torch.load(args.ckpt, map_location=dev, weights_only=False).to(dev)
    _, base_acc = evaluate(base, eval_test, dev)
    base_params = C.count(base)
    print(f"baseline: {base_params:,} params, test acc ~{base_acc:.4f}\n", flush=True)

    rows = []
    for D in args.bonds:
        model = torch.load(args.ckpt, map_location=dev, weights_only=False).to(dev)
        model.lstm = LinearLSTM.from_nn_lstm(model.lstm).to(dev)
        ih = model.lstm.ih
        mpo, rel = mpoify_linear(ih, D, n_cores=3, warm_start=True)
        model.lstm.ih = mpo.to(dev)
        _, acc_pre = evaluate(model, eval_test, dev)

        ft_train = C.cap(train_loader, args.train_subset, args.batch_size, shuffle=True)
        fit(model, ft_train, ft_val, epochs=args.finetune_epochs,
            lr=args.finetune_lr, weight_decay=1e-5, device=dev,
            ckpt_path=f"checkpoints/deeplob_ih_bond{D}.pt", log_every=0)
        model.load_state_dict(torch.load(f"checkpoints/deeplob_ih_bond{D}.pt",
                                         map_location=dev))
        _, acc_post = evaluate(model, eval_test, dev)

        rows.append((D, mpo.n_params(), C.count(model), acc_pre, acc_post))
        print(f"bond {D:2d} | ih params {mpo.n_params():5d} | model {C.count(model):,} | "
              f"acc no-retrain ~{acc_pre:.4f} | acc retrained ~{acc_post:.4f}", flush=True)

    print(f"\n{'bond':>4} {'ih params':>10} {'model params':>13} "
          f"{'no-retrain':>11} {'retrained':>10}")
    print("-" * 54)
    print(f"{'dense':>4} {49408:>10} {base_params:>13} {'--':>11} {base_acc:>10.4f}")
    for D, ihp, mp, pre, post in rows:
        print(f"{D:>4} {ihp:>10} {mp:>13} {pre:>11.4f} {post:>10.4f}")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
