"""DeepLOB + MPO compression: a small, modular toolkit.

Modules:
  config    Config dataclass (data / model / train / mpo hyperparameters)
  data      FI-2010 loading, windowing, DataLoaders (+ synthetic smoke loaders)
  models    DeepLOB CNN+LSTM backbone + configurable FCN head (MPO target)
  mpo       MPOLinear: an nn.Linear whose weight is a matrix product operator
  compress  Post-hoc replacement of trained Linear layers with MPOLinear
  engine    train / evaluate / fit loops
"""
from .config import Config
from .mpo import MPOLinear, MPOConv2d, factorize
from .models import DeepLOBNet, LinearLSTM, build_model, count_parameters
from .compress import compress_model, mpoify_linear, format_report
from .engine import classification_metrics, fit, evaluate, train_one_epoch

__all__ = [
    "Config", "MPOLinear", "MPOConv2d", "factorize", "DeepLOBNet", "LinearLSTM",
    "build_model", "count_parameters", "compress_model", "mpoify_linear",
    "format_report", "fit", "evaluate", "train_one_epoch", "classification_metrics",
]
