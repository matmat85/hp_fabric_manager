from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}.learned"


@dataclass
class LearnedModel:
    """Simple linear model: power ≈ bias + k * offset_degC"""
    k_w_per_deg: float
    bias_w: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LearnedModel":
        return cls(
            k_w_per_deg=float(d.get("k_w_per_deg", 0.0)),
            bias_w=float(d.get("bias_w", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"k_w_per_deg": self.k_w_per_deg, "bias_w": self.bias_w}


class LearnedStore:
    def __init__(self, hass: HomeAssistant) -> None:
        self._store = Store(hass, _STORAGE_VERSION, _STORAGE_KEY)
        self._data: dict[str, dict[str, Any]] = {}

    async def async_load(self) -> None:
        data = await self._store.async_load()
        if isinstance(data, dict):
            self._data = data
        else:
            self._data = {}

    async def async_save(self) -> None:
        await self._store.async_save(self._data)

    def get_zone_model(self, zone_id: str) -> LearnedModel | None:
        z = self._data.get(zone_id)
        if not isinstance(z, dict):
            return None
        return LearnedModel.from_dict(z)

    def set_zone_model(self, zone_id: str, model: LearnedModel) -> None:
        self._data[zone_id] = model.to_dict()
