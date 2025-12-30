from __future__ import annotations

from typing import Any

from homeassistant.components.number import NumberEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import HpfmCoordinator


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    coord: HpfmCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    ents: list[NumberEntity] = []
    for zone_id in coord.zones:
        ents.append(TargetRoomTempNumber(entry, coord, zone_id))
    async_add_entities(ents)


class _BaseZoneNumber(NumberEntity):
    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, coord: HpfmCoordinator, zone_id: str) -> None:
        self._entry = entry
        self._coord = coord
        self._zone_id = zone_id

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=self._entry.title,
        )


class TargetRoomTempNumber(_BaseZoneNumber):
    _attr_native_min_value = 15.0
    _attr_native_max_value = 25.0
    _attr_native_step = 0.5
    _attr_unit_of_measurement = "°C"

    def __init__(self, entry: ConfigEntry, coord: HpfmCoordinator, zone_id: str) -> None:
        super().__init__(entry, coord, zone_id)
        z = coord.zones[zone_id]
        self._attr_unique_id = f"{entry.entry_id}_{zone_id}_target_room_temp"
        self._attr_name = f"{z.cfg.zone_name} Target Room Temp"

    @property
    def native_value(self) -> float:
        return float(self._coord.zones[self._zone_id].rt.target_room_temp)

    async def async_set_native_value(self, value: float) -> None:
        self._coord.zones[self._zone_id].rt.target_room_temp = float(value)
        self.async_write_ha_state()


