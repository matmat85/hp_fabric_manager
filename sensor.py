from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, ATTR_ZONE
from .coordinator import HpfmCoordinator


@dataclass(frozen=True)
class _SensorDesc:
    key: str
    name_suffix: str
    unit: str | None
    device_class: SensorDeviceClass | None = None


SENSORS = [
    _SensorDesc("room_temp_c", "Room Temp", "°C", SensorDeviceClass.TEMPERATURE),
    _SensorDesc("temp_error_c", "Temp Error", "°C", SensorDeviceClass.TEMPERATURE),
    _SensorDesc("outdoor_temp_c", "Outdoor Temp", "°C", SensorDeviceClass.TEMPERATURE),
    _SensorDesc("wind_speed", "Wind Speed", None, None),  # Units vary by source
    _SensorDesc("actual_power_w", "Actual Power", "W", SensorDeviceClass.POWER),
    _SensorDesc("predicted_power_w", "Predicted Power", "W", SensorDeviceClass.POWER),
    _SensorDesc("power_state", "Power State", None, None),
    _SensorDesc("learned_k_w_per_deg", "Learned K", "W/°C", None),
    _SensorDesc("learned_bias_w", "Learned Bias", "W", SensorDeviceClass.POWER),
    _SensorDesc("climate_setpoint_c", "Heat Pump Setpoint", "°C", SensorDeviceClass.TEMPERATURE),
    _SensorDesc("desired_setpoint_c", "Desired Setpoint", "°C", SensorDeviceClass.TEMPERATURE),
    _SensorDesc("ramped_setpoint_c", "Ramped Setpoint", "°C", SensorDeviceClass.TEMPERATURE),
    _SensorDesc("control_status", "Control Status", None, None),
    _SensorDesc("learn_updates", "Learn Updates", None, None),
    _SensorDesc("model_status", "Model Status", None, None),
    _SensorDesc("last_learn_at", "Last Learn", None, SensorDeviceClass.TIMESTAMP),
    _SensorDesc("last_learn_offset_c", "Last Learn Offset", "°C", SensorDeviceClass.TEMPERATURE),
    _SensorDesc("last_learn_power_w", "Last Learn Power", "W", SensorDeviceClass.POWER),
    # Kick/warmup debug telemetry
    _SensorDesc("kick_setpoint_c", "Kick Setpoint", "°C", SensorDeviceClass.TEMPERATURE),
    _SensorDesc("kick_debug_waited_s", "Kick Debug Waited Seconds", "s", None),
    _SensorDesc("kick_debug_since_last_kick_s", "Kick Debug Since Last Kick", "s", None),
    _SensorDesc("last_action_was_kick", "Last Action Was Kick", None, None),
    _SensorDesc("warmup_hold_until", "Warmup Hold Until", None, SensorDeviceClass.TIMESTAMP),
]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    coord: HpfmCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]

    ents: list[SensorEntity] = []
    for zone_id, zone in coord.zones.items():
        for sd in SENSORS:
            ents.append(HpfmZoneSensor(entry, coord, zone_id, sd))
    async_add_entities(ents)


class HpfmZoneSensor(SensorEntity):
    _attr_has_entity_name = True
    # Zones update themselves, but they do not push state changes to entities.
    # Let Home Assistant poll these sensors so UI values refresh.
    _attr_should_poll = True

    def __init__(self, entry: ConfigEntry, coord: HpfmCoordinator, zone_id: str, sd: _SensorDesc) -> None:
        self._entry = entry
        self._coord = coord
        self._zone_id = zone_id
        self._sd = sd

        z = coord.zones[zone_id]
        self._attr_unique_id = f"{entry.entry_id}_{zone_id}_{sd.key}"
        self._attr_name = f"{z.cfg.zone_name} {sd.name_suffix}"
        self._attr_native_unit_of_measurement = sd.unit
        self._attr_device_class = sd.device_class

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=self._entry.title,
        )

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {ATTR_ZONE: self._zone_id}

    @property
    def native_value(self) -> Any:
        z = self._coord.zones[self._zone_id]
        if self._sd.key == "learned_k_w_per_deg":
            return round(z.rt.model.k_w_per_deg, 2)
        if self._sd.key == "learned_bias_w":
            return round(z.rt.model.bias_w, 1)
        if self._sd.key == "learn_updates":
            return int(getattr(z.rt, "learn_updates", 0) or 0)
        if self._sd.key == "last_learn_at":
            return getattr(z.rt, "last_learn_at", None)
        # Kick debug sensors
        if self._sd.key == "kick_debug_next_sp":
            v = getattr(z.rt, "kick_debug_next_sp", None)
            if v is not None:
                return round(v, 2)
            return None
        if self._sd.key == "kick_debug_waited_s":
            v = getattr(z.rt, "kick_debug_waited_s", None)
            if v is not None:
                return int(v)
            return None
        if self._sd.key == "kick_debug_cooldown_s":
            v = getattr(z.rt, "kick_debug_cooldown_s", None)
            if v is not None:
                return int(v)
            return None
        v = getattr(z.rt, self._sd.key, None)
        if isinstance(v, float):
            return round(v, 2)
        return v

    async def async_update(self) -> None:
        # zones update themselves; nothing required
        return
