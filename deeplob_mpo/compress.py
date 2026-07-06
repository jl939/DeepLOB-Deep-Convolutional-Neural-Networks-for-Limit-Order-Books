"""Post-hoc MPO compression: replace trained nn.Linear / nn.Conv2d layers (and,
optionally, the fused nn.LSTM gate matrices) with MPO factorizations,
warm-started from the trained weights via tensor-train SVD.

A layer is only replaced when the MPO is actually smaller than the dense layer
(the `max_ratio` guard) -- so tiny or poorly-factorizable matrices, where MPO
would *grow* the layer, are skipped automatically."""
from __future__ import annotations
from dataclasses import dataclass
import torch.nn as nn

from .mpo import MPOLinear, MPOConv2d, factorize
from .models import LinearLSTM


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


def _dense_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _matrix_shape(m: nn.Module):
    """Effective (in, out) of the weight matrix MPO would factorize, or None
    for module types we do not compress."""
    if isinstance(m, nn.Linear):
        return m.in_features, m.out_features
    if isinstance(m, nn.Conv2d):
        # the kernel reshaped to (out_channels, in_channels*kH*kW) is the matrix
        return m.in_channels * m.kernel_size[0] * m.kernel_size[1], m.out_channels
    return None


def _build_candidate(m: nn.Module, bond_dim, n_cores, warm_start):
    """Return (mpo_module, mpo_params, rel_err) for a compressible leaf."""
    if isinstance(m, nn.Linear):
        mpo, rel = mpoify_linear(m, bond_dim, n_cores, warm_start)
        return mpo, mpo.n_params(), rel
    if isinstance(m, nn.Conv2d):
        mpo, rel = MPOConv2d.from_conv2d(m, bond_dim, n_cores, warm_start)
        return mpo, mpo.n_params(), rel
    return None


def compress_model(model, bond_dim, n_cores=3, min_dim=64, warm_start=True,
                   include=None, max_ratio=None, compress_conv=False,
                   compress_lstm=False):
    """In-place replace eligible layers with their MPO factorization.

    By default only ``nn.Linear`` layers are considered. Set ``compress_conv``
    to also factorize ``nn.Conv2d`` kernels (via ``MPOConv2d``), and
    ``compress_lstm`` to first rewrite every ``nn.LSTM`` as a ``LinearLSTM`` so
    its input->gate (``ih``) and hidden->gate (``hh``) matrices become ordinary
    ``nn.Linear`` targets.

    A layer is compressed only when it passes *both*:
      - ``min(in, out) >= min_dim``  (cheap size pre-filter), and
      - ``mpo_params < dense_params * max_ratio``  (the ratio guard), if
        ``max_ratio`` is not None -- this skips matrices MPO would enlarge
        (tiny or prime-factored dims), so "compress everything" stays safe.

    If ``include`` is given (a set/list of layer names), only those layers are
    considered. Returns a list of LayerReport for the layers actually replaced.
    """
    include = set(include) if include is not None else None

    # Expose the fused LSTM gate matrices as nn.Linear so they can be compressed
    # by the ordinary Linear pass below. Done first so ih/hh appear in the scan.
    if compress_lstm:
        for name, m in list(model.named_modules()):
            if isinstance(m, nn.LSTM):
                parent, attr = _get_parent(model, name)
                setattr(parent, attr, LinearLSTM.from_nn_lstm(m))

    types = (nn.Linear,) + ((nn.Conv2d,) if compress_conv else ())
    targets = []
    for name, m in model.named_modules():
        if not isinstance(m, types):
            continue
        shape = _matrix_shape(m)
        if shape is None or min(shape) < min_dim:
            continue
        if include is not None and name not in include:
            continue
        targets.append((name, m))

    reports = []
    for name, m in targets:
        dense = _dense_params(m)
        mpo, mpo_params, rel = _build_candidate(m, bond_dim, n_cores, warm_start)
        if max_ratio is not None and mpo_params >= dense * max_ratio:
            continue                     # MPO would not shrink this layer; skip
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, mpo)
        reports.append(LayerReport(name, dense, mpo_params, rel))
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
