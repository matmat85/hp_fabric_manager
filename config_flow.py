from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.selector import EntitySelector, EntitySelectorConfig

from .const import DOMAIN, DEFAULT_SCAN_INTERVAL_SECONDS


class HpfmConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return HpfmOptionsFlow(config_entry)

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema(
                    {
                        vol.Optional("name", default="Heat Pump Fabric Manager"): str,
                        vol.Optional("scan_interval_seconds", default=DEFAULT_SCAN_INTERVAL_SECONDS): vol.All(
                            vol.Coerce(int), vol.Range(min=10, max=300)
                        ),
                    }
                ),
            )

        title = user_input.get("name") or "Heat Pump Fabric Manager"
        options = {"scan_interval_seconds": int(user_input.get("scan_interval_seconds", DEFAULT_SCAN_INTERVAL_SECONDS))}
        return self.async_create_entry(title=title, data={"zones": {}}, options=options)

    async def async_step_import(self, user_input: dict[str, Any]):
        return await self.async_step_user(user_input)


class HpfmOptionsFlow(config_entries.OptionsFlow):
    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        # Newer Home Assistant exposes config_entry as a read-only property.
        # Store the entry on the underlying attribute that the property reads.
        self._config_entry = config_entry
        self._edit_zone_id: str | None = None

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        return await self.async_step_menu()

    async def async_step_menu(self, user_input: dict[str, Any] | None = None):
        zones = self.config_entry.data.get("zones", {})
        menu_options: dict[str, str] = {
            "add_zone": "Add zone",
            "modify_settings": "Modify settings",
        }
        if zones:
            menu_options["edit_zone"] = "Edit zone"

        return self.async_show_menu(
            step_id="menu",
            menu_options=menu_options,
        )

    async def async_step_edit_zone(self, user_input: dict[str, Any] | None = None):
        zones: dict[str, Any] = dict(self.config_entry.data.get("zones", {}))
        zone_options = {zone_id: zd.get("zone_name", zone_id) for zone_id, zd in zones.items()}

        if user_input is None:
            return self.async_show_form(
                step_id="edit_zone",
                data_schema=vol.Schema({vol.Required("zone_id"): vol.In(zone_options)}),
            )

        self._edit_zone_id = str(user_input["zone_id"])
        return await self.async_step_edit_zone_details()

    async def async_step_edit_zone_details(self, user_input: dict[str, Any] | None = None):
        zones: dict[str, Any] = dict(self.config_entry.data.get("zones", {}))
        zone_id = self._edit_zone_id
        if not zone_id or zone_id not in zones:
            return await self.async_step_menu()

        zd = zones[zone_id]

        if user_input is None:
            return self.async_show_form(
                step_id="edit_zone_details",
                data_schema=vol.Schema(
                    {
                        vol.Required("climate_entity", default=zd.get("climate_entity")): EntitySelector(
                            EntitySelectorConfig(domain=["climate"], multiple=False)
                        ),
                        vol.Optional("room_temp_sensors", default=list(zd.get("room_temp_sensors", []))): EntitySelector(
                            EntitySelectorConfig(domain=["sensor"], multiple=True)
                        ),
                        vol.Optional("power_sensor", default=zd.get("power_sensor")): EntitySelector(
                            EntitySelectorConfig(domain=["sensor"], multiple=False)
                        ),
                        vol.Optional("outdoor_temp_sensor", default=zd.get("outdoor_temp_sensor")): EntitySelector(
                            EntitySelectorConfig(domain=["sensor"], multiple=False)
                        ),
                        vol.Optional(
                            "outdoor_humidity_sensor", default=zd.get("outdoor_humidity_sensor")
                        ): EntitySelector(EntitySelectorConfig(domain=["sensor"], multiple=False)),
                        vol.Optional("wind_speed_sensor", default=zd.get("wind_speed_sensor")): EntitySelector(
                            EntitySelectorConfig(domain=["sensor"], multiple=False)
                        ),
                        vol.Optional("enabled", default=bool(zd.get("enabled", True))): bool,
                        vol.Optional("target_room_temp", default=float(zd.get("target_room_temp", 20.0))): vol.All(
                            vol.Coerce(float), vol.Range(min=15.0, max=25.0)
                        ),
                        vol.Optional("min_offset_c", default=float(zd.get("min_offset_c", 0.2))): vol.All(
                            vol.Coerce(float), vol.Range(min=0.0, max=2.0)
                        ),
                        vol.Optional("max_offset_c", default=float(zd.get("max_offset_c", 3.0))): vol.All(
                            vol.Coerce(float), vol.Range(min=1.0, max=6.0)
                        ),
                    }
                ),
            )

        # Update the zone in-place (keep zone_id and zone_name stable)
        zd = dict(zd)
        zd["climate_entity"] = user_input["climate_entity"]
        zd["room_temp_sensors"] = list(user_input.get("room_temp_sensors", []))
        zd["power_sensor"] = user_input.get("power_sensor")
        zd["outdoor_temp_sensor"] = user_input.get("outdoor_temp_sensor")
        zd["outdoor_humidity_sensor"] = user_input.get("outdoor_humidity_sensor")
        zd["wind_speed_sensor"] = user_input.get("wind_speed_sensor")
        zd["enabled"] = bool(user_input.get("enabled", True))
        zd["target_room_temp"] = float(user_input.get("target_room_temp", 20.0))
        zd["min_offset_c"] = float(user_input.get("min_offset_c", 0.2))
        zd["max_offset_c"] = float(user_input.get("max_offset_c", 3.0))

        zones[zone_id] = zd
        self.hass.config_entries.async_update_entry(self.config_entry, data={"zones": zones})
        await self.hass.config_entries.async_reload(self.config_entry.entry_id)
        self._edit_zone_id = None
        return self.async_create_entry(title="", data={})

    async def async_step_add_zone(self, user_input: dict[str, Any] | None = None):
        if user_input is None:
            return self.async_show_form(
                step_id="add_zone",
                data_schema=vol.Schema(
                    {
                        vol.Required("zone_name"): str,
                        vol.Required("climate_entity"): EntitySelector(
                            EntitySelectorConfig(domain=["climate"], multiple=False)
                        ),
                        vol.Optional("room_temp_sensors", default=[]): EntitySelector(
                            EntitySelectorConfig(domain=["sensor"], multiple=True)
                        ),
                        vol.Optional("power_sensor"): EntitySelector(
                            EntitySelectorConfig(domain=["sensor"], multiple=False)
                        ),
                        vol.Optional("outdoor_temp_sensor"): EntitySelector(
                            EntitySelectorConfig(domain=["sensor"], multiple=False)
                        ),
                        vol.Optional("outdoor_humidity_sensor"): EntitySelector(
                            EntitySelectorConfig(domain=["sensor"], multiple=False)
                        ),
                        vol.Optional("wind_speed_sensor"): EntitySelector(
                            EntitySelectorConfig(domain=["sensor"], multiple=False)
                        ),
                        vol.Optional("enabled", default=True): bool,
                        vol.Optional("target_room_temp", default=20.0): vol.All(
                            vol.Coerce(float), vol.Range(min=15.0, max=25.0)
                        ),
                        vol.Optional("min_offset_c", default=0.2): vol.All(
                            vol.Coerce(float), vol.Range(min=0.0, max=2.0)
                        ),
                        vol.Optional("max_offset_c", default=3.0): vol.All(
                            vol.Coerce(float), vol.Range(min=1.0, max=6.0)
                        ),
                    }
                ),
            )

        zones = dict(self.config_entry.data.get("zones", {}))

        # Create zone id (stable-ish)
        base = (user_input["zone_name"] or "zone").strip().lower().replace(" ", "_")
        zone_id = base
        i = 2
        while zone_id in zones:
            zone_id = f"{base}_{i}"
            i += 1

        zones[zone_id] = {
            "zone_name": user_input["zone_name"],
            "climate_entity": user_input["climate_entity"],
            "room_temp_sensors": list(user_input.get("room_temp_sensors", [])),
            "power_sensor": user_input.get("power_sensor"),
            "outdoor_temp_sensor": user_input.get("outdoor_temp_sensor"),
            "outdoor_humidity_sensor": user_input.get("outdoor_humidity_sensor"),
            "wind_speed_sensor": user_input.get("wind_speed_sensor"),
            "enabled": bool(user_input.get("enabled", True)),
            "target_room_temp": float(user_input.get("target_room_temp", 20.0)),
            "min_offset_c": float(user_input.get("min_offset_c", 0.2)),
            "max_offset_c": float(user_input.get("max_offset_c", 3.0)),
        }

        self.hass.config_entries.async_update_entry(self.config_entry, data={"zones": zones})
        await self.hass.config_entries.async_reload(self.config_entry.entry_id)
        return self.async_create_entry(title="", data={})

    async def async_step_modify_settings(self, user_input: dict[str, Any] | None = None):
        if user_input is None:
            current_interval = self.config_entry.options.get("scan_interval_seconds", DEFAULT_SCAN_INTERVAL_SECONDS)
            return self.async_show_form(
                step_id="modify_settings",
                data_schema=vol.Schema(
                    {
                        vol.Optional("scan_interval_seconds", default=current_interval): vol.All(
                            vol.Coerce(int), vol.Range(min=10, max=300)
                        ),
                    }
                ),
            )

        return self.async_create_entry(title="", data=user_input)
