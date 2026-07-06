"""Models for FI-2010. All MPO compression targets are plain nn.Linear layers,
so deeplob_mpo.compress works on every architecture here unchanged.

  - DeepLOBNet     : CNN+LSTM backbone + wide FCN head  (head = MPO target)
  - MLPNet         : flatten window -> wide MLP         (first Linear = MPO target)
  - TransformerNet : encoder over timesteps            (attn + FFN = MPO targets)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLOBBackbone(nn.Module):
    """CNN + Inception + LSTM feature extractor from Zhang et al. (DeepLOB).

    Input  : (B, 1, T, 40)
    Output : (B, lstm_hidden) feature vector (last LSTM timestep).

    The kernel shapes are *structural* -- (1,2)/(1,2)/(1,10) walk the 40-column
    LOB layout (pair price+size, merge ask/bid, then merge all 10 levels) and
    (4,1)/(3,1)/(5,1) are the temporal windows. Only the *widths* are tunable:
      conv_ch       channels in the 3 conv blocks
      inception_ch  channels per inception branch (LSTM input = 3 * inception_ch)
      lstm_hidden   LSTM hidden size = output feature width
    Scaling these up enlarges the MPO targets (LSTM gates, inception convs)
    while keeping the architecture faithful to the paper (defaults = paper).
    """

    def __init__(self, conv_ch=32, inception_ch=64, lstm_hidden=64):
        super().__init__()
        c, e = conv_ch, inception_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c, (1, 2), (1, 2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(c))
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, (1, 2), (1, 2)), nn.Tanh(), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (4, 1)), nn.Tanh(), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (4, 1)), nn.Tanh(), nn.BatchNorm2d(c))
        self.conv3 = nn.Sequential(
            nn.Conv2d(c, c, (1, 10)), nn.LeakyReLU(0.01), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(c),
            nn.Conv2d(c, c, (4, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(c))
        self.inp1 = nn.Sequential(
            nn.Conv2d(c, e, (1, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(e),
            nn.Conv2d(e, e, (3, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(e))
        self.inp2 = nn.Sequential(
            nn.Conv2d(c, e, (1, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(e),
            nn.Conv2d(e, e, (5, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(e))
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), (1, 1), (1, 0)),
            nn.Conv2d(c, e, (1, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(e))
        self.lstm = nn.LSTM(input_size=3 * e, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x, _ = self.lstm(x)                       # h0/c0 default to zeros
        return x[:, -1, :]                        # (B, 64)


class FCNHead(nn.Module):
    """Stack of fully-connected layers -> logits. These Linear layers are the
    layers replaced by MPOLinear during compression."""

    def __init__(self, in_dim=64, hidden=512, depth=2, n_classes=3, dropout=0.1):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU(0.01),
                       nn.BatchNorm1d(hidden)]
            if dropout:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepLOBNet(nn.Module):
    """Backbone + FCN head. Returns logits (use nn.CrossEntropyLoss)."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = DeepLOBBackbone(cfg.conv_channels,
                                        cfg.inception_channels, cfg.lstm_hidden)
        self.head = FCNHead(cfg.lstm_hidden, cfg.head_hidden, cfg.head_depth,
                            cfg.n_classes, cfg.head_dropout)

    def forward(self, x):
        return self.head(self.backbone(x))


class LinearLSTM(nn.Module):
    """Single-layer, batch_first LSTM built from explicit nn.Linear projections,
    so the input->hidden matrix (`ih`) can be replaced by MPOLinear and trained.

    Drop-in for nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True):
    same forward signature (x, (h0, c0)) and same (output, (h_n, c_n)) return.
    Runs as a Python loop over timesteps, so it is slower than cuDNN's nn.LSTM.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, 4 * hidden_size)   # weight_ih + bias_ih
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)  # weight_hh + bias_hh

    @classmethod
    def from_nn_lstm(cls, lstm: nn.LSTM) -> "LinearLSTM":
        """Copy weights from a trained nn.LSTM (num_layers=1). Lossless."""
        m = cls(lstm.input_size, lstm.hidden_size)
        with torch.no_grad():
            m.ih.weight.copy_(lstm.weight_ih_l0)
            m.ih.bias.copy_(lstm.bias_ih_l0)
            m.hh.weight.copy_(lstm.weight_hh_l0)
            m.hh.bias.copy_(lstm.bias_hh_l0)
        return m

    def forward(self, x, hx=None):
        B, T, _ = x.shape
        if hx is None:
            h = x.new_zeros(B, self.hidden_size)
            c = x.new_zeros(B, self.hidden_size)
        else:
            h, c = hx[0][0], hx[1][0]
        outs = []
        for t in range(T):
            g = self.ih(x[:, t]) + self.hh(h)           # PyTorch gate order: i,f,g,o
            i, f, cell, o = g.chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            c = f * c + i * torch.tanh(cell)
            h = o * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs, dim=1), (h.unsqueeze(0), c.unsqueeze(0))


class ResidualBlock(nn.Module):
    """Pre-activation residual FC block; its Linear(h, h) is an MPO target."""

    def __init__(self, h, dropout):
        super().__init__()
        self.norm = nn.BatchNorm1d(h)
        self.fc = nn.Linear(h, h)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.fc(F.leaky_relu(self.norm(x), 0.01)))


class MLPNet(nn.Module):
    """Flatten the (T x 40) window and run a deep, wide residual MLP. The stem
    Linear(T*40 -> hidden) (~millions of params) and every residual block's
    Linear(hidden -> hidden) are large MPO targets. Input: (B, 1, T, 40)."""

    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.window_T * 40
        h = cfg.mlp_hidden
        self.stem = nn.Linear(in_dim, h)
        self.blocks = nn.Sequential(
            *[ResidualBlock(h, cfg.head_dropout) for _ in range(cfg.mlp_blocks)])
        self.norm = nn.BatchNorm1d(h)
        self.head = nn.Linear(h, cfg.n_classes)

    def forward(self, x):
        x = self.stem(x.flatten(1))
        x = self.blocks(x)
        return self.head(F.leaky_relu(self.norm(x), 0.01))


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with explicit Linear q/k/v/o, so the
    attention projections (not just the FFN) are MPO-compressible — unlike
    nn.MultiheadAttention, whose in_proj is a raw Parameter."""

    def __init__(self, d_model, n_heads, ff_mult, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, ff_mult * d_model)
        self.ff2 = nn.Linear(ff_mult * d_model, d_model)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _attn(self, x):
        B, T, _ = x.shape
        shp = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = shp(self.q(x)), shp(self.k(x)), shp(self.v(x))
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, T, self.h * self.dh)
        return self.o(a)

    def forward(self, x):
        x = x + self.drop(self._attn(self.n1(x)))
        h = self.ff2(self.drop(F.gelu(self.ff1(self.n2(x)))))
        return x + self.drop(h)


class TransformerNet(nn.Module):
    """Linear-embed the 40 LOB features, add a learned positional code, run a
    stack of encoder layers over the T timesteps, mean-pool -> logits.
    Input: (B, 1, T, 40)."""

    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Linear(40, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.window_T, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([
            EncoderLayer(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.head_dropout)
            for _ in range(cfg.tf_depth)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.fc = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, x):
        x = x.squeeze(1)                          # (B, T, 40)
        x = self.embed(x) + self.pos
        for layer in self.layers:
            x = layer(x)
        return self.fc(self.norm(x).mean(dim=1))  # mean-pool over time


_REGISTRY = {"deeplob": DeepLOBNet, "mlp": MLPNet, "transformer": TransformerNet}


def build_model(cfg) -> nn.Module:
    torch.manual_seed(cfg.seed)
    if cfg.model not in _REGISTRY:
        raise ValueError(f"unknown model {cfg.model!r}; choose from {list(_REGISTRY)}")
    return _REGISTRY[cfg.model](cfg)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
