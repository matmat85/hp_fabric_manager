from __future__ import annotations

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import HpfmCoordinator


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    coord: HpfmCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    ents: list[SwitchEntity] = []
    for zone_id in coord.zones:
        ents.append(ZoneEnableSwitch(entry, coord, zone_id))
    async_add_entities(ents)


class ZoneEnableSwitch(SwitchEntity):
    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, entry: ConfigEntry, coord: HpfmCoordinator, zone_id: str) -> None:
        self._entry = entry
        self._coord = coord
        self._zone_id = zone_id

        z = coord.zones[zone_id]
        self._attr_unique_id = f"{entry.entry_id}_{zone_id}_enable"
        self._attr_name = f"{z.cfg.zone_name} Enable"

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry.entry_id)},
            name=self._entry.title,
        )

    @property
    def is_on(self) -> bool:
        return bool(self._coord.zones[self._zone_id].rt.enabled)

    async def async_turn_on(self, **kwargs) -> None:
        self._coord.zones[self._zone_id].rt.enabled = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs) -> None:
        self._coord.zones[self._zone_id].rt.enabled = False
        self.async_write_ha_state()
