#!/usr/bin/env python
"""Train a baseline FI-2010 model (mlp / transformer / deeplob).

Saves <out>.pt (state_dict) and <out>_config.json. The companion script
scripts/compress.py loads these to MPO-compress the trained model.

Examples
--------
  # quick pipeline check, no data / no GPU
  python scripts/train.py --model mlp --smoke

  # real run on a GPU
  python scripts/train.py --model mlp --device mps --epochs 50
  python scripts/train.py --model transformer --device cuda --epochs 50 --d-model 128
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeplob_mpo import Config, build_model, count_parameters, fit, evaluate
from deeplob_mpo.data import build_loaders, build_smoke_loaders


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("model")
    g.add_argument("--model", default="mlp",
                   choices=["mlp", "transformer", "deeplob"])
    g.add_argument("--mlp-hidden", type=int, default=None)
    g.add_argument("--mlp-blocks", type=int, default=None)
    g.add_argument("--d-model", type=int, default=None)
    g.add_argument("--n-heads", type=int, default=None)
    g.add_argument("--tf-depth", type=int, default=None)
    g.add_argument("--ff-mult", type=int, default=None)
    g.add_argument("--head-hidden", type=int, default=None)
    g.add_argument("--head-depth", type=int, default=None)
    g.add_argument("--dropout", dest="head_dropout", type=float, default=None)

    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default="jupyter_pytorch")
    g.add_argument("--horizon", dest="horizon_k", type=int, default=None,
                   help="prediction-horizon index 0..4")
    g.add_argument("--window", dest="window_T", type=int, default=None)

    g = p.add_argument_group("train")
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--weight-decay", type=float, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--device", default="cpu")

    g = p.add_argument_group("io")
    g.add_argument("--out", default=None, help="default checkpoints/<model>.pt")
    g.add_argument("--log-every", type=int, default=1)
    g.add_argument("--smoke", action="store_true",
                   help="tiny synthetic data + 2 epochs to verify the pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_overrides(
        model=args.model, mlp_hidden=args.mlp_hidden, mlp_blocks=args.mlp_blocks,
        d_model=args.d_model, n_heads=args.n_heads, tf_depth=args.tf_depth,
        ff_mult=args.ff_mult, head_hidden=args.head_hidden,
        head_depth=args.head_depth, head_dropout=args.head_dropout,
        data_dir=args.data_dir, horizon_k=args.horizon_k, window_T=args.window_T,
        epochs=2 if args.smoke else args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        device=args.device)

    out = args.out or f"checkpoints/{cfg.model}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    train_loader, val_loader, test_loader = (
        build_smoke_loaders(cfg) if args.smoke else build_loaders(cfg))

    model = build_model(cfg)
    print(f"model={cfg.model}  params={count_parameters(model):,}  device={cfg.device}")

    res = fit(model, train_loader, val_loader, epochs=cfg.epochs, lr=cfg.lr,
              weight_decay=cfg.weight_decay, device=cfg.device,
              ckpt_path=out, log_every=args.log_every)

    model.load_state_dict(torch.load(out, map_location=cfg.device))
    _, test_acc = evaluate(model, test_loader, cfg.device)
    print(f"\nbest val acc: {res['best_val_acc']:.4f}   test acc: {test_acc:.4f}")

    with open(os.path.splitext(out)[0] + "_config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"saved checkpoint -> {out}")


if __name__ == "__main__":
    main()
