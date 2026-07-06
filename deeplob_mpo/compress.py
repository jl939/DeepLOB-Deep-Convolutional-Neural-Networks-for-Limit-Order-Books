"""Post-hoc MPO compression: replace trained nn.Linear layers with MPOLinear,
warm-started from the trained weights via tensor-train SVD."""
from __future__ import annotations
from dataclasses import dataclass
import torch.nn as nn

from .mpo import MPOLinear, factorize


@dataclass
class LayerReport:
    name: str
    dense_params: int
    mpo_params: int
    rel_err: float                       # TT-SVD reconstruction error of W

    @property
    def ratio(self) -> float:
        return self.mpo_params / self.dense_params


def _get_parent(model, dotted):
    """Return (parent_module, attr_name) for a (possibly top-level) dotted path."""
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def mpoify_linear(linear: nn.Linear, bond_dim, n_cores=3, warm_start=True):
    """Build an MPOLinear matching a given nn.Linear, optionally warm-started."""
    inf, outf = linear.in_features, linear.out_features
    mpo = MPOLinear(inf, outf,
                    in_shape=factorize(inf, n_cores),
                    out_shape=factorize(outf, n_cores),
                    bond_dim=bond_dim, bias=linear.bias is not None)
    rel = 0.0
    if warm_start:
        rel = mpo.init_from_weight(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None)
    return mpo, rel


def compress_model(model, bond_dim, n_cores=3, min_dim=64, warm_start=True,
                   include=None):
    """In-place replace eligible nn.Linear layers with MPOLinear.

    Eligible = min(in_features, out_features) >= min_dim (skips tiny heads).
    If `include` is given (a set/list of layer names), only those layers are
    compressed. Returns a list of LayerReport.
    """
    include = set(include) if include is not None else None
    targets = [(name, m) for name, m in model.named_modules()
               if isinstance(m, nn.Linear)
               and min(m.in_features, m.out_features) >= min_dim
               and (include is None or name in include)]

    reports = []
    for name, linear in targets:
        dense = linear.weight.numel() + (
            linear.bias.numel() if linear.bias is not None else 0)
        mpo, rel = mpoify_linear(linear, bond_dim, n_cores, warm_start)
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, mpo)
        reports.append(LayerReport(name, dense, mpo.n_params(), rel))
    return reports


def format_report(reports, total_params_before, total_params_after) -> str:
    lines = [f"{'layer':<22}{'dense':>10}{'mpo':>10}{'ratio':>8}{'TT-SVD err':>12}",
             "-" * 62]
    for r in reports:
        lines.append(f"{r.name:<22}{r.dense_params:>10}{r.mpo_params:>10}"
                     f"{r.ratio:>8.3f}{r.rel_err:>12.3f}")
    lines.append("-" * 62)
    lines.append(f"{'MODEL TOTAL':<22}{total_params_before:>10}"
                 f"{total_params_after:>10}"
                 f"{total_params_after / total_params_before:>8.3f}")
    return "\n".join(lines)
