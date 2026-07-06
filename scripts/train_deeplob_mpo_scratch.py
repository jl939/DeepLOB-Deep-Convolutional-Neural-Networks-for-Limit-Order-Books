#!/usr/bin/env python
"""Train DeepLOB with an MPO LSTM-input matrix FROM SCRATCH (random init, no
SVD) -- exactly the paper's method (Gao et al.): MPO layers are randomly
initialized and the whole network is trained from zero.

Also measures the reconstruction norm ||W_dense - W_MPO|| / ||W_dense|| against
the trained *dense* DeepLOB, to answer "is that norm big for the from-scratch
(paper-style) approach too?"

Example:
  python scripts/train_deeplob_mpo_scratch.py --bond 8 --epochs 25 --device mps
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
from deeplob_mpo.mpo import MPOLinear, factorize
from deeplob_mpo.config import Config
from deeplob_mpo.data import build_loaders
from deeplob_mpo.engine import evaluate, fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bond", type=int, default=8)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-subset", type=int, default=40000)
    p.add_argument("--eval-subset", type=int, default=10000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev, D = args.device, args.bond

    cfg = Config(device=dev, batch_size=args.batch_size)
    train_loader, val_loader, test_loader = build_loaders(cfg)
    ft_val = C.cap(val_loader, args.eval_subset, args.batch_size)
    eval_test = C.cap(test_loader, args.eval_subset, args.batch_size)

    # dense reference: the trained DeepLOB's LSTM input matrix
    ref = torch.load("jupyter_pytorch/best_val_model_pytorch",
                     map_location="cpu", weights_only=False)
    W_dense = ref.lstm.weight_ih_l0.data.cpu().clone()

    # build DeepLOB from scratch (all random), LSTM input matrix = random MPO
    torch.manual_seed(0)
    model = C.deeplob(3)
    model.lstm = LinearLSTM(192, 64)                       # random init, not from dense
    model.lstm.ih = MPOLinear(192, 256, in_shape=factorize(192, 3),
                              out_shape=factorize(256, 3), bond_dim=D)  # random cores
    model.to(dev)

    def recon_norm():
        Wm = model.lstm.ih.reconstruct_weight().detach().cpu()
        return ((Wm - W_dense).norm() / W_dense.norm()).item()

    print(f"MPO params in lstm.ih: {model.lstm.ih.n_params()} (dense would be {W_dense.numel()})")
    print(f"||W_dense - W_MPO|| / ||W_dense|| at RANDOM INIT (untrained): {recon_norm():.3f}\n",
          flush=True)

    fit(model, C.cap(train_loader, args.train_subset, args.batch_size, shuffle=True),
        ft_val, epochs=args.epochs, lr=args.lr, weight_decay=1e-5, device=dev,
        ckpt_path="checkpoints/deeplob_mpo_scratch.pt")
    model.load_state_dict(torch.load("checkpoints/deeplob_mpo_scratch.pt", map_location=dev))

    _, acc = evaluate(model, test_loader, dev)
    print(f"\nfrom-scratch MPO DeepLOB: full-test acc {acc:.4f}", flush=True)
    print(f"||W_dense - W_MPO|| / ||W_dense|| AFTER from-scratch training: {recon_norm():.3f}",
          flush=True)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()
