"""Training / evaluation loops (cleaned-up version of the notebook's batch_gd)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.long)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    losses, preds, tgts = [], [], []
    for x, y in loader:
        x, y = x.to(device, dtype=torch.float), y.to(device, dtype=torch.long)
        out = model(x)
        if criterion is not None:
            losses.append(criterion(out, y).item())
        preds.append(out.argmax(1).cpu().numpy())
        tgts.append(y.cpu().numpy())
    preds, tgts = np.concatenate(preds), np.concatenate(tgts)
    acc = float((preds == tgts).mean())
    loss = float(np.mean(losses)) if losses else float("nan")
    return loss, acc


def fit(model, train_loader, val_loader, *, epochs, lr, weight_decay, device,
        ckpt_path=None, log_every=1):
    """Train, tracking best val accuracy. Saves best state_dict to ckpt_path."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    best_acc, history = 0.0, []
    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
        if log_every and ep % log_every == 0:
            print(f"epoch {ep:3d}  train_loss {tr_loss:.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
                  f"{'  *' if val_acc == best_acc else ''}")
    return {"best_val_acc": best_acc, "history": history}
