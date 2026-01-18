"""
Muon Optimizer - Momentum with Newton-Schulz Orthogonalization.

Muon is a variant of SGD with Momentum that applies a Newton-Schulz iteration
to approximately orthogonalize each gradient update. This leads to faster
convergence on transformer training tasks.

Reference: https://kellerjordan.github.io/posts/muon/
"""

import torch
from torch import Tensor
from torch.optim import Optimizer


def newton_schulz_orthogonalize(G: Tensor, steps: int = 5) -> Tensor:
    """
    Approximate matrix orthogonalization via Newton-Schulz iteration.

    Given a matrix G, computes an approximation of G @ (G.T @ G)^(-1/2),
    which has orthonormal columns if G has more rows than columns,
    or orthonormal rows if G has more columns than rows.

    Args:
        G: Input gradient matrix
        steps: Number of Newton-Schulz iterations (5 is usually sufficient)

    Returns:
        Approximately orthogonalized matrix
    """
    assert G.ndim >= 2

    # Coefficients for the quintic iteration (faster convergence)
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Work in bfloat16 for efficiency
    X = G.bfloat16()

    # Handle non-square matrices: work with the smaller dimension
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Normalize to prevent numerical issues
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if needed
    if transposed:
        X = X.mT

    return X


class Muon(Optimizer):
    """
    Muon Optimizer: Momentum with Newton-Schulz Orthogonalization.

    Muon applies Newton-Schulz orthogonalization to gradient updates,
    which empirically leads to faster convergence on transformer training.
    It's particularly effective for the dense matrices in attention and MLP layers.

    Best practices:
    - Use Muon for transformer weight matrices (attention Q/K/V/O, MLP)
    - Use AdamW for embeddings and layer norms
    - Recommended lr=0.02, momentum=0.95

    Args:
        params: Model parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)

    Reference: https://kellerjordan.github.io/posts/muon/
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)

        # Group parameters by size for efficient batched operations
        params_list = list(params)
        param_groups = []
        for size in {p.numel() for p in params_list}:
            group_params = [p for p in params_list if p.numel() == size]
            param_groups.append(dict(params=group_params))

        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # Initialize momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]

                # Update momentum buffer: buf = momentum * buf + (1 - momentum) * g
                buf.lerp_(g, 1 - momentum)

                # Optionally apply Nesterov momentum
                if nesterov:
                    g = g.lerp(buf, momentum)
                else:
                    g = buf

                # Apply Newton-Schulz orthogonalization
                g = newton_schulz_orthogonalize(g, steps=ns_steps)

                # Update parameters with scaling based on matrix shape
                # This helps balance updates across different-shaped matrices
                scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g, alpha=-lr * scale)

        return loss
