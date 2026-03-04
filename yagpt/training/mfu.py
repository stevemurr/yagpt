"""
Model FLOPs Utilization (MFU) Callback.

Estimates how efficiently the GPU is being used by comparing actual throughput
to theoretical peak throughput.

MFU = (model_flops_per_step / step_time) / peak_gpu_flops

For a transformer: flops_per_token ~= 6 * N (forward + backward)
"""

import time
from typing import TYPE_CHECKING

from .callbacks import Callback, TrainState

if TYPE_CHECKING:
    from .trainer import Trainer


# Peak BF16 TFLOPS for common GPUs
GPU_PEAK_TFLOPS = {
    "dgx_spark": 213,   # NVIDIA DGX Spark (Blackwell, BF16)
    "h100_sxm": 989,    # H100 SXM (BF16)
    "h100_pcie": 756,   # H100 PCIe (BF16)
    "a100_sxm": 312,    # A100 SXM 80GB (BF16)
    "a100_pcie": 312,   # A100 PCIe 80GB (BF16)
    "rtx_4090": 165,    # RTX 4090 (BF16)
}


class MFUCallback(Callback):
    """
    Estimates Model FLOPs Utilization each step.

    Uses the approximation: flops_per_token = 6 * num_params
    (accounts for forward + backward pass).

    Args:
        num_params: Number of model parameters (excluding embeddings)
        peak_tflops: Peak GPU throughput in TFLOPS (BF16). Defaults to DGX Spark.
    """

    def __init__(self, num_params: int, peak_tflops: float = 213.0):
        self.num_params = num_params
        self.peak_flops = peak_tflops * 1e12  # Convert to FLOPS
        self.last_time: float | None = None

    def on_train_start(self, trainer: "Trainer") -> None:
        self.last_time = time.time()

    def on_step_end(self, trainer: "Trainer", state: TrainState) -> None:
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            return

        dt = now - self.last_time
        self.last_time = now

        if dt <= 0:
            return

        # Total tokens processed this step
        batch_tokens = trainer.config.total_batch_size

        # Approximate FLOPs: 6 * N * tokens (forward + backward)
        step_flops = 6 * self.num_params * batch_tokens

        # MFU = actual_flops / peak_flops
        state.mfu = step_flops / (dt * self.peak_flops)
