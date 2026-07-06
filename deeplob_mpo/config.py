"""Central configuration for training and MPO compression."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch as _torch


def _detect_device() -> str:
    if _torch.cuda.is_available():
        return "cuda"
    if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DEFAULT_DEVICE = _detect_device()


@dataclass
class Config:
    # -- data (FI-2010) --------------------------------------------------------
    data_dir: str = "jupyter_pytorch"   # dir holding the extracted *.txt files
    horizon_k: int = 4                  # prediction-horizon index, 0..4
    window_T: int = 100                 # look-back length
    n_classes: int = 3

    # -- model -----------------------------------------------------------------
    model: str = "deeplob"              # "deeplob" | "mlp" | "transformer"

    # deeplob: CNN+LSTM backbone (64-d feature) + wide FCN head (MPO target)
    head_hidden: int = 512
    head_depth: int = 2                 # number of hidden Linear layers
    head_dropout: float = 0.1

    # mlp: flatten the (T x 40) window -> deep residual MLP. The stem
    # (T*40 -> mlp_hidden) and every residual block (mlp_hidden^2) are large
    # MPO targets. Wide + deep so it is a real predictor, not a toy FCN.
    mlp_hidden: int = 1024
    mlp_blocks: int = 4

    # transformer: encoder over the T timesteps; attention + FFN Linears are
    # the MPO targets.
    d_model: int = 128
    n_heads: int = 4
    tf_depth: int = 2
    ff_mult: int = 4

    # -- training --------------------------------------------------------------
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5          # L2 reg, the paper's alpha term (Eq. 7)
    device: str = _DEFAULT_DEVICE       # "cuda" / "mps" / "cpu"
    seed: int = 0

    # -- MPO compression -------------------------------------------------------
    bond_dim: int = 16
    mpo_n_cores: int = 3
    mpo_min_dim: int = 64               # only compress Linear with min(in,out) >= this
    finetune_epochs: int = 10
    finetune_lr: float = 5e-5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_overrides(cls, **kwargs) -> "Config":
        """Build a Config, applying only the keyword args that are not None."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in kwargs.items() if k in known and v is not None})
