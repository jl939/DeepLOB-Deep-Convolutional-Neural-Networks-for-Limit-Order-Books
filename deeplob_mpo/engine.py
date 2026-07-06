"""Training / evaluation loops (cleaned-up version of the notebook's batch_gd)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def _is_cuda(device) -> bool:
    return str(device).startswith("cuda")


def classification_metrics(preds, tgts, n_classes=None):
    """Dependency-free classification metrics for FI-2010's class labels."""
    preds = np.asarray(preds)
    tgts = np.asarray(tgts)
    if n_classes is None:
        pred_max = preds.max() if preds.size else 0
        tgt_max = tgts.max() if tgts.size else 0
        n_classes = int(max(pred_max, tgt_max) + 1)

    accuracy = float((preds == tgts).mean()) if len(tgts) else float("nan")
    precisions, recalls, f1s, supports = [], [], [], []
    for cls in range(n_classes):
        pred_pos = preds == cls
        true_pos = tgts == cls
        tp = float(np.logical_and(pred_pos, true_pos).sum())
        fp = float(np.logical_and(pred_pos, ~true_pos).sum())
        fn = float(np.logical_and(~pred_pos, true_pos).sum())
        support = float(true_pos.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    supports = np.asarray(supports, dtype=float)
    weights = supports / supports.sum() if supports.sum() else np.zeros_like(supports)
    denom = float(((tgts - tgts.mean()) ** 2).sum()) if len(tgts) else 0.0
    r2 = (1.0 - float(((tgts - preds) ** 2).sum()) / denom
          if denom else (1.0 if np.array_equal(preds, tgts) else 0.0))

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
        "precision_weighted": float(np.dot(weights, precisions)),
        "recall_weighted": float(np.dot(weights, recalls)),
        "f1_weighted": float(np.dot(weights, f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "r2": float(r2),
    }


def train_one_epoch(model, loader, optimizer, criterion, device,
                    return_metrics=False, scaler=None):
    model.train()
    losses, preds, tgts = [], [], []
    n_classes = None
    amp_ctx = torch.autocast("cuda") if scaler is not None else torch.autocast("cpu", enabled=False)
    for x, y in loader:
        x = x.to(device, dtype=torch.float, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad()
        with amp_ctx:
            out = model(x)
            n_classes = out.shape[1]
            loss = criterion(out, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        if return_metrics:
            preds.append(out.detach().argmax(1).cpu().numpy())
            tgts.append(y.detach().cpu().numpy())
    loss = float(np.mean(losses))
    if not return_metrics:
        return loss
    preds, tgts = np.concatenate(preds), np.concatenate(tgts)
    return loss, classification_metrics(preds, tgts, n_classes)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, return_metrics=False):
    model.eval()
    losses, preds, tgts = [], [], []
    n_classes = None
    amp_ctx = torch.autocast("cuda") if _is_cuda(device) else torch.autocast("cpu", enabled=False)
    for x, y in loader:
        x = x.to(device, dtype=torch.float, non_blocking=True)
        y = y.to(device, dtype=torch.long, non_blocking=True)
        with amp_ctx:
            out = model(x)
            n_classes = out.shape[1]
            if criterion is not None:
                losses.append(criterion(out, y).item())
        preds.append(out.argmax(1).cpu().numpy())
        tgts.append(y.cpu().numpy())
    preds, tgts = np.concatenate(preds), np.concatenate(tgts)
    metrics = classification_metrics(preds, tgts, n_classes)
    acc = metrics["accuracy"]
    loss = float(np.mean(losses)) if losses else float("nan")
    if return_metrics:
        return loss, metrics
    return loss, acc


def _monitor_value(row: dict, monitor: str) -> tuple[float, bool]:
    if monitor not in row:
        valid = ", ".join(sorted(row))
        raise ValueError(f"unknown monitor {monitor!r}; choose one of: {valid}")
    value = row[monitor]
    if not np.isfinite(value):
        raise ValueError(f"monitor {monitor!r} is not finite: {value}")
    return float(value), monitor.endswith("loss")


def fit(model, train_loader, val_loader, *, epochs, lr, weight_decay, device,
        adam_eps=1e-8, ckpt_path=None, log_every=1, metrics_logger=None,
        monitor="val_acc", early_stopping_patience=None, min_delta=0.0,
        label_smoothing=0.0):
    """Train, tracking the best validation metric.

    Saves the best state_dict to ckpt_path when provided. The default monitor
    preserves the original behavior: maximize validation accuracy.
    """
    raw_model = model.to(device)
    train_model = raw_model
    if _is_cuda(device):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "compile"):
            train_model = torch.compile(raw_model)
    scaler = torch.amp.GradScaler("cuda") if _is_cuda(device) else None
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr,
                                 weight_decay=weight_decay, eps=adam_eps)
    best_acc, best_score, best_epoch, history = float("-inf"), None, None, []
    epochs_without_improvement = 0
    for ep in range(1, epochs + 1):
        tr_loss, train_metrics = train_one_epoch(
            train_model, train_loader, optimizer, criterion, device,
            return_metrics=True, scaler=scaler)
        val_loss, val_metrics = evaluate(
            train_model, val_loader, device, criterion, return_metrics=True)
        val_acc = val_metrics["accuracy"]
        row = {"epoch": ep, "train_loss": tr_loss,
               "val_loss": val_loss, "val_acc": val_acc}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        score, lower_is_better = _monitor_value(row, monitor)
        if best_score is None:
            improved = True
        elif lower_is_better:
            improved = score < best_score - min_delta
        else:
            improved = score > best_score + min_delta
        if metrics_logger:
            metrics_logger({
                "epoch": ep,
                "train/loss": tr_loss,
                "val/loss": val_loss,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            })
        if improved:
            best_score = score
            best_epoch = ep
            epochs_without_improvement = 0
            if ckpt_path:
                torch.save(raw_model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
        best_acc = max(best_acc, val_acc)
        if log_every and ep % log_every == 0:
            print(f"epoch {ep:3d}  train_loss {tr_loss:.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
                  f"{'  *' if improved else ''}")
        if (early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience):
            print(f"early stopping at epoch {ep}; best {monitor} "
                  f"{best_score:.4f} at epoch {best_epoch}")
            break
    return {
        "best_val_acc": best_acc,
        "best_monitor": monitor,
        "best_monitor_value": best_score,
        "best_epoch": best_epoch,
        "history": history,
    }
