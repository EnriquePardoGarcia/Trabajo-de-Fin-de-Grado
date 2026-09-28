import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration import CODE_DIM, IN_DIM


class LinearLISTAEncoder(nn.Module):
    """LISTA encoder (unrolled ISTA, Gregor & LeCun 2010) with a fixed num_iters steps.

        B    = W_e * x                     (W_e has no bias)
        Z(0) = B                           (no ReLU)
        Z(t) = ReLU(B + S * Z(t-1))        t = 1 ... num_iters-1

    W_e projects the input into code space; S iteratively refines it and
    implements "explaining away". The learnable shrinkage threshold is S's
    bias: ReLU(B + W_S*z + b_S) is equivalent to a per-atom threshold (-b_S)
    learned jointly with the rest of the network, with no separate parameter.
    """

    def __init__(self, in_dim=IN_DIM, code_dim=CODE_DIM, num_iters=10):
        super().__init__()
        self.in_dim    = in_dim
        self.code_dim  = code_dim
        self.num_iters = num_iters

        # W_e: dictionary (input -> code projection), WITHOUT bias.
        # S: iterative refinement; its bias is the learnable shrinkage threshold.
        self.W_e = nn.Linear(in_dim, code_dim, bias=False)
        self.S   = nn.Linear(code_dim, code_dim, bias=True)

        # S is initialized to zero: the first iteration is Z(1)=ReLU(B), a simple
        # projection that eases initial convergence. "Explaining away" (competition
        # between atoms) is learned by S from there.
        nn.init.zeros_(self.S.weight)
        if self.S.bias is not None:
            nn.init.zeros_(self.S.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B = fixed projection of the input (does not change across iterations)
        B = self.W_e(x)
        z = B  # Z(0) = B (no ReLU)
        # Each iteration refines z by applying ReLU to B + S(z).
        # -S.bias acts as the per-atom shrinkage threshold.
        for _ in range(self.num_iters - 1):
            z = F.relu(B + self.S(z))
        return z


class LinearLISTADecoder(nn.Module):
    """Decoder tied to the encoder: reconstructs using the transpose of W_e as the dictionary."""

    def __init__(self, encoder: LinearLISTAEncoder):
        super().__init__()
        self.encoder = encoder
        self.b_d = None  # no bias in the decoder (centered reconstruction)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Shared dictionary: W_d = W_e^T  (tied weights)
        W_e = self.encoder.W_e.weight
        W_d = W_e.t()
        x_hat = F.linear(z, W_d)
        return x_hat
