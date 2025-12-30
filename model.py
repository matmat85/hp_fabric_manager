from __future__ import annotations

from dataclasses import dataclass

from .const import EMA_ALPHA


@dataclass
class LinearPowerModel:
    """power_w ≈ bias_w + k_w_per_deg * offset_c"""
    k_w_per_deg: float
    bias_w: float

    def predict(self, offset_c: float) -> float:
        return self.bias_w + self.k_w_per_deg * offset_c

    def update_ema(self, offset_c: float, power_w: float) -> None:
        """
        Update model using EMA:
        We treat k as (power - bias)/offset, and bias as (power - k*offset).
        """
        if offset_c <= 0:
            return

        # Estimate k from this sample
        sample_k = (power_w - self.bias_w) / offset_c

        # Bound k to sane range (avoid wild jumps from noise)
        sample_k = max(0.0, min(sample_k, 2000.0))

        self.k_w_per_deg = (1 - EMA_ALPHA) * self.k_w_per_deg + EMA_ALPHA * sample_k

        # Re-estimate bias from updated k
        sample_bias = power_w - self.k_w_per_deg * offset_c
        sample_bias = max(-500.0, min(sample_bias, 1500.0))
        self.bias_w = (1 - EMA_ALPHA) * self.bias_w + EMA_ALPHA * sample_bias
