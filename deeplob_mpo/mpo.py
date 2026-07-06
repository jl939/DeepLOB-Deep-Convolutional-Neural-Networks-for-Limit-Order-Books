"""
MPO (Matrix Product Operator) linear layer.

Reference: Gao et al., "Compressing deep neural networks by matrix product
operators", Phys. Rev. Research 2, 023300 (2020).

A dense linear map  y = W x + b   with  W of shape (Ny, Nx)  is replaced by an
MPO factorization (Eq. 3-5):

    Nx = prod(in_shape)  = I_1 * I_2 * ... * I_n
    Ny = prod(out_shape) = J_1 * J_2 * ... * J_n
    W[j1..jn, i1..in] = Tr( w1[j1,i1] ... wn[jn,in] )

Each local tensor w_k has shape (D_{k-1}, J_k, I_k, D_k); D_0 = D_n = 1.
Only the local-tensor elements are stored/trained. Param count (Eq. 6):
    sum_k  D_{k-1} * J_k * I_k * D_k   (+ Ny for the bias).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def factorize(n: int, k: int) -> list[int]:
    """Split integer n into k factors as evenly as possible (descending)."""
    factors, remaining = [], n
    for i in range(k, 0, -1):
        target = round(remaining ** (1.0 / i))
        f = next((d for d in range(target, 0, -1) if remaining % d == 0), 1)
        factors.append(f)
        remaining //= f
    assert int(np.prod(factors)) == n, (n, k, factors)
    return factors


class MPOLinear(nn.Module):
    """Drop-in replacement for nn.Linear whose weight is stored as an MPO."""

    def __init__(self, in_features, out_features, in_shape=None,
                 out_shape=None, bond_dim=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if in_shape is None:
            in_shape = factorize(in_features, 3)
        if out_shape is None:
            out_shape = factorize(out_features, len(in_shape))
        assert len(in_shape) == len(out_shape), "need same number of cores"
        assert int(np.prod(in_shape)) == in_features
        assert int(np.prod(out_shape)) == out_features

        self.in_shape = list(in_shape)
        self.out_shape = list(out_shape)
        self.n = len(in_shape)

        # bond dims clamped to the feasible TT-rank at each cut, so a requested
        # D larger than the achievable rank cannot mismatch SVD-truncated cores.
        site = [out_shape[k] * in_shape[k] for k in range(self.n)]
        bonds = [1]
        for k in range(1, self.n):
            left = int(np.prod(site[:k]))
            right = int(np.prod(site[k:]))
            bonds.append(min(bond_dim, left, right))
        bonds.append(1)
        self.bonds = bonds

        self.cores = nn.ParameterList()
        for k in range(self.n):
            core = torch.empty(bonds[k], out_shape[k], in_shape[k], bonds[k + 1])
            fan_in = bonds[k] * in_shape[k]
            nn.init.normal_(core, std=1.0 / np.sqrt(fan_in))
            self.cores.append(nn.Parameter(core))

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    # -- introspection ---------------------------------------------------------
    def n_params(self) -> int:
        n = sum(c.numel() for c in self.cores)
        return n + (self.bias.numel() if self.bias is not None else 0)

    def reconstruct_weight(self) -> torch.Tensor:
        """Contract the cores into the dense (Ny, Nx) matrix W."""
        w = self.cores[0].reshape(self.out_shape[0], self.in_shape[0], self.bonds[1])
        for k in range(1, self.n):
            w = torch.tensordot(w, self.cores[k], dims=([-1], [0]))
        w = w.squeeze(-1)
        n = self.n
        perm = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2))
        return w.permute(*perm).contiguous().reshape(self.out_features, self.in_features)

    # -- warm start from a trained dense weight (post-hoc compression) ---------
    @torch.no_grad()
    def init_from_weight(self, W: torch.Tensor, bias: torch.Tensor | None = None) -> float:
        """Initialize the cores from a trained dense weight via tensor-train
        SVD, truncating each bond. Returns the relative Frobenius
        reconstruction error ||W_mpo - W|| / ||W||."""
        n, Is, Js = self.n, self.in_shape, self.out_shape
        # SVD runs on CPU (MPS has no linalg.svd); copy_ moves cores back to the
        # parameters' device, so this works regardless of the model's device.
        Wc = W.detach().to("cpu", torch.float32)
        T = Wc.reshape(*Js, *Is)
        perm = []
        for k in range(n):
            perm += [k, n + k]
        T = T.permute(*perm).contiguous().reshape(*[Js[k] * Is[k] for k in range(n)])

        cores, Dl = [], 1
        for k in range(n - 1):
            M = T.reshape(Dl * Js[k] * Is[k], -1)
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            Dr = min(self.bonds[k + 1], S.shape[0])
            U, S, Vh = U[:, :Dr], S[:Dr], Vh[:Dr]
            cores.append(U.reshape(Dl, Js[k], Is[k], Dr))
            T = torch.diag(S) @ Vh
            Dl = Dr
        cores.append(T.reshape(Dl, Js[n - 1], Is[n - 1], 1))

        for p, c in zip(self.cores, cores):
            p.copy_(c.to(p.dtype))            # copy_ handles CPU -> p.device
        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.to(self.bias.dtype))
        recon = self.reconstruct_weight()
        Wd = W.to(recon.device, recon.dtype)
        return ((recon - Wd).norm() / Wd.norm()).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x = x.reshape(-1, self.in_features)
        B = x.shape[0]
        carry = x.reshape(B, 1, 1, self.in_shape[0], -1)
        for k in range(self.n):
            carry = torch.einsum('bpdir,djie->bpjer', carry, self.cores[k])
            B_, P, J, Dn, R = carry.shape
            if k < self.n - 1:
                I_next = self.in_shape[k + 1]
                carry = carry.reshape(B_, P * J, Dn, I_next, R // I_next)
            else:
                carry = carry.reshape(B_, P * J, Dn, 1, R)
        out = carry.reshape(B, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*lead, self.out_features)


class MPOConv2d(nn.Module):
    """Conv2d whose kernel is stored as an MPO. A convolution is a linear map
    from a flattened patch (in_channels*kH*kW) to out_channels, so the kernel
    reshaped to (out_channels, in_channels*kH*kW) is an ordinary weight matrix
    -- represented here by an internal MPOLinear. Each forward rebuilds the
    kernel and runs a standard conv2d (saves storage, not compute)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bond_dim=4, n_cores=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size)
        self.stride = stride
        self.padding = padding
        in_features = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.mpo = MPOLinear(in_features, out_channels,
                             in_shape=factorize(in_features, n_cores),
                             out_shape=factorize(out_channels, n_cores),
                             bond_dim=bond_dim, bias=bias)

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, bond_dim, n_cores=3, warm_start=True):
        m = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                conv.stride, conv.padding, bond_dim, n_cores,
                bias=conv.bias is not None)
        rel = 0.0
        if warm_start:
            W = conv.weight.data.reshape(conv.out_channels, -1)   # (out, in*kH*kW)
            rel = m.mpo.init_from_weight(
                W, conv.bias.data if conv.bias is not None else None)
        return m, rel

    def n_params(self) -> int:
        return self.mpo.n_params()

    def forward(self, x):
        W = self.mpo.reconstruct_weight().reshape(
            self.out_channels, self.in_channels, *self.kernel_size)
        return F.conv2d(x, W, self.mpo.bias, self.stride, self.padding)
