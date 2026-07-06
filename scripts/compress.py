#!/usr/bin/env python
"""MPO-compress a trained model, then fine-tune to recover accuracy.

Loads checkpoints/<model>.pt + its _config.json (written by train.py), replaces
the eligible nn.Linear layers with warm-started MPOLinear, and reports the
parameter/accuracy trade-off before and after a short fine-tune.

Examples
--------
  python scripts/compress.py --smoke
  python scripts/compress.py --ckpt checkpoints/mlp.pt --bond-dim 8 --device mps
  python scripts/compress.py --ckpt checkpoints/transformer.pt --bond-dim 16 --finetune-epochs 10
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeplob_mpo import (Config, build_model, count_parameters, compress_model,
                         format_report, fit, evaluate)
from deeplob_mpo.data import build_loaders, build_smoke_loaders


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/mlp.pt")
    p.add_argument("--bond-dim", type=int, default=16, help="MPO compression knob")
    p.add_argument("--n-cores", type=int, default=3)
    p.add_argument("--min-dim", type=int, default=64,
                   help="only compress Linear with min(in,out) >= this")
    p.add_argument("--only", default=None,
                   help="comma-separated layer names to compress (e.g. 'stem'); "
                        "default compresses all eligible layers")
    p.add_argument("--list-layers", action="store_true",
                   help="print the compressible layer names and exit")
    p.add_argument("--no-warm-start", action="store_true",
                   help="random-init the MPO cores instead of TT-SVD warm start")
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--finetune-lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=None,
                   help="override batch size for fine-tune (larger = faster/steadier)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="checkpoints/compressed.pt")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = os.path.splitext(args.ckpt)[0] + "_config.json"
    cfg = (Config.from_dict(json.load(open(cfg_path)))
           if os.path.isfile(cfg_path) else Config())
    cfg.device = args.device
    if args.batch_size:
        cfg.batch_size = args.batch_size

    train_loader, val_loader, test_loader = (
        build_smoke_loaders(cfg) if args.smoke else build_loaders(cfg))

    # 1. load the trained baseline
    model = build_model(cfg).to(cfg.device)
    if os.path.isfile(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=cfg.device))
    elif not args.smoke:
        raise FileNotFoundError(f"{args.ckpt} not found; run scripts/train.py first.")
    before = count_parameters(model)

    if args.list_layers:
        import torch.nn as nn
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                print(f"{name:20s} {tuple(m.weight.shape)}  {m.weight.numel():,} params")
        return

    include = [s.strip() for s in args.only.split(",")] if args.only else None

    # 2. reference accuracy
    _, acc_ref = evaluate(model, test_loader, cfg.device)

    # 3. MPO-compress the eligible Linear layers
    reports = compress_model(model, args.bond_dim, args.n_cores, args.min_dim,
                             warm_start=not args.no_warm_start, include=include)
    model.to(cfg.device)
    after = count_parameters(model)
    if not reports:
        print("no eligible Linear layers (try lowering --min-dim).")
        return

    # 4. accuracy right after compression (no fine-tuning)
    _, acc_compressed = evaluate(model, test_loader, cfg.device)

    print(f"\nmodel={cfg.model}  bond_dim={args.bond_dim}  "
          f"warm_start={not args.no_warm_start}")
    print(format_report(reports, before, after))
    print(f"\nbaseline test acc        : {acc_ref:.4f}")
    print(f"after MPO (no fine-tune) : {acc_compressed:.4f}")

    # 5. fine-tune the compressed model
    if args.finetune_epochs > 0:
        fit(model, train_loader, val_loader,
            epochs=2 if args.smoke else args.finetune_epochs,
            lr=args.finetune_lr, weight_decay=cfg.weight_decay,
            device=cfg.device, ckpt_path=args.out)
        model.load_state_dict(torch.load(args.out, map_location=cfg.device))
        _, acc_final = evaluate(model, test_loader, cfg.device)
        print(f"after fine-tune          : {acc_final:.4f}")

    print(f"\ncompression: {before:,} -> {after:,} params "
          f"({after / before:.1%} kept, {before / max(after, 1):.1f}x smaller)")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
