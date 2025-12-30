from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import DEFAULT_SCAN_INTERVAL_SECONDS
from .storage import LearnedStore, LearnedModel
from .zone import HeatPumpZone, ZoneConfig, ZoneRuntime

_LOGGER = logging.getLogger(__name__)


class HpfmCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, store: LearnedStore) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=f"HPFM {entry.title}",
            update_interval=None,
        )
        self.entry = entry
        self.store = store
        self.zones: dict[str, HeatPumpZone] = {}

    async def async_configure_zones(self) -> None:
        data = self.entry.data
        zones = data.get("zones", {})
        for zone_id, zd in zones.items():
            cfg = ZoneConfig(
                zone_name=zd["zone_name"],
                climate_entity=zd["climate_entity"],
                room_temp_sensors=list(zd.get("room_temp_sensors", [])),
                power_sensor=zd.get("power_sensor"),
                outdoor_temp_sensor=zd.get("outdoor_temp_sensor"),
                outdoor_humidity_sensor=zd.get("outdoor_humidity_sensor"),
                wind_speed_sensor=zd.get("wind_speed_sensor"),
            )

            rt = ZoneRuntime(
                enabled=zd.get("enabled", True),
                target_room_temp=float(zd.get("target_room_temp", 20.0)),
                max_offset_c=float(zd.get("max_offset_c", 3.0)),
                min_offset_c=float(zd.get("min_offset_c", 0.2)),
            )

            learned = self.store.get_zone_model(zone_id)
            if learned:
                rt.model.k_w_per_deg = learned.k_w_per_deg
                rt.model.bias_w = learned.bias_w

            self.zones[zone_id] = HeatPumpZone(self.hass, self.entry.entry_id, zone_id, cfg, rt)

    async def async_start(self) -> None:
        interval = int(self.entry.options.get("scan_interval_seconds", DEFAULT_SCAN_INTERVAL_SECONDS))
        for z in self.zones.values():
            await z.async_start(interval)

    async def async_stop(self) -> None:
        for z in self.zones.values():
            await z.async_stop()

    async def async_persist_learning(self) -> None:
        for zone_id, z in self.zones.items():
            self.store.set_zone_model(
                zone_id,
                LearnedModel(k_w_per_deg=z.rt.model.k_w_per_deg, bias_w=z.rt.model.bias_w),
            )
        await self.store.async_save()

    async def _async_update_data(self) -> dict[str, Any]:
        # We don't use polling updates; zones tick themselves.
        # This is here only if something triggers refresh.
        result: dict[str, Any] = {}
        for zone_id, z in self.zones.items():
            result[zone_id] = {
                "zone_name": z.cfg.zone_name,
                "enabled": z.rt.enabled,
                "target_room_temp": z.rt.target_room_temp,
                "room_temp_c": z.rt.room_temp_c,
                "climate_setpoint_c": z.rt.climate_setpoint_c,
                "climate_mode": z.rt.climate_mode,
                "fan_mode": z.rt.fan_mode,
                "predicted_power_w": z.rt.predicted_power_w,
                "learned_k_w_per_deg": z.rt.model.k_w_per_deg,
                "learned_bias_w": z.rt.model.bias_w,
            }
        return result
