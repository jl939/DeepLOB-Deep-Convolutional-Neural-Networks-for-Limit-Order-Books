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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeplob_mpo import (Config, build_model, count_parameters, compress_model,
                         format_report, fit, evaluate)
from deeplob_mpo.data import build_loaders, build_smoke_loaders
from deeplob_mpo.wandb_utils import add_wandb_args, init_wandb, log_metrics


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", default="checkpoints/mlp.pt")
    p.add_argument("--bond-dim", type=int, default=16, help="MPO compression knob")
    p.add_argument("--n-cores", type=int, default=3)
    p.add_argument("--min-dim", type=int, default=64,
                   help="only compress layers with min(in,out) >= this")
    p.add_argument("--max-ratio", type=float, default=1.0,
                   help="ratio guard: only replace a layer if mpo_params < "
                        "max_ratio * dense_params (skips layers MPO would grow; "
                        "set >1 to disable, e.g. 2.0)")
    p.add_argument("--compress-conv", action="store_true",
                   help="also factorize nn.Conv2d kernels (via MPOConv2d)")
    p.add_argument("--compress-lstm", action="store_true",
                   help="also compress the LSTM: rewrite nn.LSTM as LinearLSTM "
                        "so its ih/hh gate matrices become MPO targets")
    p.add_argument("--only", default=None,
                   help="comma-separated layer names to compress (e.g. 'stem'); "
                        "default compresses all eligible layers")
    p.add_argument("--list-layers", action="store_true",
                   help="print the layers that would be compressed (honoring "
                        "--bond-dim / --compress-conv / --compress-lstm) and exit")
    p.add_argument("--no-warm-start", action="store_true",
                   help="random-init the MPO cores instead of TT-SVD warm start")
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--finetune-lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=None,
                   help="override batch size for fine-tune (larger = faster/steadier)")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="checkpoints/compressed.pt")
    p.add_argument("--smoke", action="store_true")
    add_wandb_args(p)
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

    include = [s.strip() for s in args.only.split(",")] if args.only else None

    if args.list_layers:
        # dry run on a copy: shows exactly what would be replaced at this
        # bond dim / guard / flags, with per-layer size and reconstruction error.
        import copy
        probe = copy.deepcopy(model).to("cpu")
        reports = compress_model(
            probe, args.bond_dim, args.n_cores, args.min_dim, warm_start=True,
            include=include, max_ratio=args.max_ratio,
            compress_conv=args.compress_conv, compress_lstm=args.compress_lstm)
        if not reports:
            print("no layers pass the size + ratio guard "
                  "(try lowering --min-dim, raising --max-ratio, or "
                  "--compress-conv / --compress-lstm).")
            return
        print(f"layers that would be compressed at bond_dim={args.bond_dim}:\n")
        print(format_report(reports, before, before -
                            sum(r.dense_params - r.mpo_params for r in reports)))
        return

    run = init_wandb(
        args, cfg, job_type="compress",
        extra_config={
            "checkpoint": args.ckpt,
            "compressed_checkpoint": args.out,
            "smoke": args.smoke,
            "bond_dim": args.bond_dim,
            "mpo_n_cores": args.n_cores,
            "mpo_min_dim": args.min_dim,
            "mpo_max_ratio": args.max_ratio,
            "compress_conv": args.compress_conv,
            "compress_lstm": args.compress_lstm,
            "warm_start": not args.no_warm_start,
            "finetune_epochs": 2 if args.smoke else args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "parameters_before": before,
        })

    # 2. reference accuracy
    criterion = nn.CrossEntropyLoss()
    ref_loss, ref_metrics = evaluate(
        model, test_loader, cfg.device, criterion, return_metrics=True)
    acc_ref = ref_metrics["accuracy"]
    log_metrics(run, "test/baseline", ref_loss, ref_metrics)

    # 3. MPO-compress the eligible layers
    reports = compress_model(
        model, args.bond_dim, args.n_cores, args.min_dim,
        warm_start=not args.no_warm_start, include=include,
        max_ratio=args.max_ratio, compress_conv=args.compress_conv,
        compress_lstm=args.compress_lstm)
    model.to(cfg.device)
    after = count_parameters(model)
    if not reports:
        print("no layers passed the size + ratio guard (try lowering --min-dim, "
              "raising --max-ratio, or adding --compress-conv / --compress-lstm).")
        if run:
            run.finish()
        return
    if run:
        run.log({
            "compression/parameters_before": before,
            "compression/parameters_after": after,
            "compression/parameters_kept": after / before,
            "compression/size_reduction": before / max(after, 1),
        })
        for report in reports:
            run.log({
                "compression/layer_dense_params": report.dense_params,
                "compression/layer_mpo_params": report.mpo_params,
                "compression/layer_ratio": report.ratio,
                "compression/layer_reconstruction_error": report.rel_err,
                "compression/layer": report.name,
            })

    # 4. accuracy right after compression (no fine-tuning)
    compressed_loss, compressed_metrics = evaluate(
        model, test_loader, cfg.device, criterion, return_metrics=True)
    acc_compressed = compressed_metrics["accuracy"]
    log_metrics(run, "test/compressed", compressed_loss, compressed_metrics)

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
            device=cfg.device, ckpt_path=args.out,
            metrics_logger=run.log if run else None)
        model.load_state_dict(torch.load(args.out, map_location=cfg.device))
        final_loss, final_metrics = evaluate(
            model, test_loader, cfg.device, criterion, return_metrics=True)
        acc_final = final_metrics["accuracy"]
        log_metrics(run, "test/finetuned", final_loss, final_metrics)
        if run:
            run.summary["finetuned_test_accuracy"] = acc_final
            run.save(args.out)
        print(f"after fine-tune          : {acc_final:.4f}")

    print(f"\ncompression: {before:,} -> {after:,} params "
          f"({after / before:.1%} kept, {before / max(after, 1):.1f}x smaller)")
    if run:
        run.summary["baseline_test_accuracy"] = acc_ref
        run.summary["compressed_test_accuracy"] = acc_compressed
        run.summary["parameters_before"] = before
        run.summary["parameters_after"] = after
        run.finish()


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
