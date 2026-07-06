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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeplob_mpo import Config, build_model, count_parameters, fit, evaluate
from deeplob_mpo.data import build_loaders, build_smoke_loaders
from deeplob_mpo.wandb_utils import add_wandb_args, init_wandb, log_metrics


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
    # deeplob backbone capacity (scales the MPO targets; defaults = paper)
    g.add_argument("--conv-channels", dest="conv_channels", type=int, default=None,
                   help="deeplob: channels in the 3 conv blocks (default 32)")
    g.add_argument("--inception-channels", dest="inception_channels", type=int,
                   default=None,
                   help="deeplob: channels per inception branch (default 64)")
    g.add_argument("--lstm-hidden", dest="lstm_hidden", type=int, default=None,
                   help="deeplob: LSTM hidden / feature width (default 64)")

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
    g.add_argument("--eps", dest="adam_eps", type=float, default=None,
                   help="ADAM epsilon (paper sets this to 1.0; default 1e-8)")
    g.add_argument("--monitor", default="val_acc",
                   choices=["val_loss", "val_acc", "val_f1_macro",
                            "val_balanced_accuracy", "val_r2"],
                   help="validation metric used for checkpoint selection")
    g.add_argument("--early-stopping-patience", type=int, default=None,
                   help="stop after this many epochs without monitor improvement")
    g.add_argument("--min-delta", type=float, default=0.0,
                   help="minimum monitor improvement required to reset patience")
    g.add_argument("--label-smoothing", type=float, default=0.0,
                   help="CrossEntropyLoss label smoothing")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--device", default=None)

    g = p.add_argument_group("io")
    g.add_argument("--out", default=None, help="default checkpoints/<model>.pt")
    g.add_argument("--log-every", type=int, default=1)
    g.add_argument("--smoke", action="store_true",
                   help="tiny synthetic data + 2 epochs to verify the pipeline")
    add_wandb_args(p)
    args = p.parse_args()
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        p.error("--early-stopping-patience must be at least 1")
    if args.min_delta < 0:
        p.error("--min-delta must be non-negative")
    if not 0 <= args.label_smoothing < 1:
        p.error("--label-smoothing must be in [0, 1)")
    return args


def main():
    args = parse_args()
    cfg = Config.from_overrides(
        model=args.model, mlp_hidden=args.mlp_hidden, mlp_blocks=args.mlp_blocks,
        d_model=args.d_model, n_heads=args.n_heads, tf_depth=args.tf_depth,
        ff_mult=args.ff_mult, head_hidden=args.head_hidden,
        head_depth=args.head_depth, head_dropout=args.head_dropout,
        conv_channels=args.conv_channels,
        inception_channels=args.inception_channels, lstm_hidden=args.lstm_hidden,
        data_dir=args.data_dir, horizon_k=args.horizon_k, window_T=args.window_T,
        epochs=2 if args.smoke else args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, adam_eps=args.adam_eps,
        seed=args.seed, device=args.device)

    out = args.out or f"checkpoints/{cfg.model}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    train_loader, val_loader, test_loader = (
        build_smoke_loaders(cfg) if args.smoke else build_loaders(cfg))

    model = build_model(cfg)
    n_params = count_parameters(model)
    print(f"model={cfg.model}  params={n_params:,}  device={cfg.device}")

    run = init_wandb(
        args, cfg, job_type="train",
        extra_config={"checkpoint": out, "smoke": args.smoke,
                      "parameters": n_params, "monitor": args.monitor,
                      "early_stopping_patience": args.early_stopping_patience,
                      "min_delta": args.min_delta,
                      "label_smoothing": args.label_smoothing})

    res = fit(model, train_loader, val_loader, epochs=cfg.epochs, lr=cfg.lr,
              weight_decay=cfg.weight_decay, adam_eps=cfg.adam_eps,
              device=cfg.device, ckpt_path=out, log_every=args.log_every,
              metrics_logger=run.log if run else None,
              monitor=args.monitor,
              early_stopping_patience=args.early_stopping_patience,
              min_delta=args.min_delta,
              label_smoothing=args.label_smoothing)

    model.load_state_dict(torch.load(out, map_location=cfg.device))
    test_loss, test_metrics = evaluate(
        model, test_loader, cfg.device, nn.CrossEntropyLoss(), return_metrics=True)
    test_acc = test_metrics["accuracy"]
    log_metrics(run, "test", test_loss, test_metrics)
    if run:
        run.summary["best_val_accuracy"] = res["best_val_acc"]
        run.summary["best_epoch"] = res["best_epoch"]
        run.summary["best_monitor"] = res["best_monitor"]
        run.summary["best_monitor_value"] = res["best_monitor_value"]
        run.summary["test_accuracy"] = test_acc
        run.summary["checkpoint"] = out
    print(f"\nbest {res['best_monitor']}: {res['best_monitor_value']:.4f} "
          f"at epoch {res['best_epoch']}   "
          f"best val acc: {res['best_val_acc']:.4f}   test acc: {test_acc:.4f}")

    cfg_out = os.path.splitext(out)[0] + "_config.json"
    with open(cfg_out, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    if run:
        run.save(out)
        run.save(cfg_out)
        run.finish()
    print(f"saved checkpoint -> {out}")


if __name__ == "__main__":
    main()
