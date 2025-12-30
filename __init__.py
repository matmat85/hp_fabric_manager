from __future__ import annotations

from datetime import timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_interval

from .const import DOMAIN, PLATFORMS
from .coordinator import HpfmCoordinator
from .storage import LearnedStore


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    store = LearnedStore(hass)
    await store.async_load()

    coord = HpfmCoordinator(hass, entry, store)
    await coord.async_configure_zones()
    await coord.async_start()

    hass.data[DOMAIN][entry.entry_id] = {"coordinator": coord, "store": store}

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Periodically persist learning (every 10 minutes).
    # Use HA's scheduler instead of an infinite task to avoid slowing startup.
    async def _persist(_now) -> None:
        await coord.async_persist_learning()

    unsub = async_track_time_interval(hass, _persist, timedelta(minutes=10))
    hass.data[DOMAIN][entry.entry_id]["persist_unsub"] = unsub

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    data = hass.data[DOMAIN].pop(entry.entry_id, None)
    if data:
        coord: HpfmCoordinator = data["coordinator"]
        unsub = data.get("persist_unsub")
        if unsub:
            unsub()
        await coord.async_stop()
        await coord.async_persist_learning()

    ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    return ok
