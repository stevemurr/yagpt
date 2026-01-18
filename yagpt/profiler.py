"""
Training profiler for monitoring data loading and compute performance.

Provides context managers for timing different phases of training and
analyzing bottlenecks.
"""

import time
from contextlib import contextmanager
from typing import Optional


class TrainingProfiler:
    """Profiles training loop performance to identify bottlenecks.

    Tracks data loading time vs compute time to help optimize training throughput.
    Provides recommendations for improving performance.

    Example:
        profiler = TrainingProfiler(enabled=True)

        for step in range(num_steps):
            with profiler.time_compute():
                # Forward and backward pass
                loss = model(x, y)
                loss.backward()

                with profiler.time_data_load():
                    x, y = next(dataloader)

            if step % log_interval == 0:
                metrics = profiler.get_metrics()
                logger.log(metrics)
    """

    def __init__(self, enabled: bool = True):
        """Initialize profiler.

        Args:
            enabled: If False, profiling is disabled (no overhead)
        """
        self.enabled = enabled
        self.data_load_times = []
        self.compute_times = []
        self._current_data_start: Optional[float] = None
        self._current_compute_start: Optional[float] = None

    @contextmanager
    def time_data_load(self):
        """Context manager for timing data loading operations.

        Usage:
            with profiler.time_data_load():
                x, y = next(dataloader)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self.data_load_times.append(time.perf_counter() - start)

    @contextmanager
    def time_compute(self):
        """Context manager for timing compute operations.

        Usage:
            with profiler.time_compute():
                loss = model(x, y)
                loss.backward()
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self.compute_times.append(time.perf_counter() - start)

    def get_metrics(self) -> dict:
        """Get profiling metrics and clear internal state.

        Returns averaged metrics over all recorded timings since last call.
        Automatically clears internal timing lists after computing metrics.

        Returns:
            Dictionary with keys:
                - data_load_ms: Average data loading time in milliseconds
                - compute_ms: Average compute time in milliseconds
                - data_compute_ratio: Ratio of data loading to compute time

            Returns empty dict if profiling is disabled.
        """
        if not self.enabled:
            return {}

        # Calculate averages
        avg_data_time = sum(self.data_load_times) / len(self.data_load_times) if self.data_load_times else 0
        avg_compute_time = sum(self.compute_times) / len(self.compute_times) if self.compute_times else 0
        data_to_compute_ratio = avg_data_time / avg_compute_time if avg_compute_time > 0 else 0

        # Clear for next interval
        self.data_load_times.clear()
        self.compute_times.clear()

        return {
            'data_load_ms': avg_data_time * 1000,
            'compute_ms': avg_compute_time * 1000,
            'data_compute_ratio': data_to_compute_ratio,
        }

    def print_analysis(self, step: int, num_workers: int, metrics: Optional[dict] = None):
        """Print performance analysis with recommendations.

        Args:
            step: Current training step
            num_workers: Number of data loading workers
            metrics: Optional pre-computed metrics dict. If None, calls get_metrics()
        """
        if not self.enabled:
            return

        # Get metrics if not provided
        if metrics is None:
            metrics = self.get_metrics()

        if not metrics:
            return

        data_load_ms = metrics['data_load_ms']
        compute_ms = metrics['compute_ms']
        ratio = metrics['data_compute_ratio']

        print(f"\n[Performance Profile - Step {step}]")
        print(f"  Data loading: {data_load_ms:.2f}ms")
        print(f"  Compute:      {compute_ms:.2f}ms")
        print(f"  Data/Compute: {ratio:.2f}x")

        # Provide recommendations based on ratio
        if ratio > 0.5:
            print(f"  ⚠️  Data loading is slow! Consider:")
            print(f"     - Pre-tokenizing dataset (see scripts/preprocess_fineweb.py)")
            print(f"     - Increasing num_workers (currently: {num_workers})")
        elif ratio < 0.1:
            print(f"  ✓ Excellent! GPU is the bottleneck (as desired)")
        print()

    def reset(self):
        """Clear all timing data."""
        self.data_load_times.clear()
        self.compute_times.clear()
