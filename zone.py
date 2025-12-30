from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from datetime import timedelta

_LOGGER = logging.getLogger(__name__)

from .const import (
    DEFAULT_K_W_PER_DEG,
    DEFAULT_BIAS_W,
    DEFAULT_MAX_OFFSET_C,
    DEFAULT_MIN_OFFSET_C,
    DEFAULT_MAX_SETPOINT_STEP_C_PER_UPDATE,
    DEFAULT_SETPOINT_MIN_C,
    DEFAULT_SETPOINT_MAX_C,
    DEFAULT_HEATING_START_HOLD_SECONDS,
    LEARN_MIN_OFFSET_C,
    LEARN_MIN_POWER_W,
    POWER_IDLE_MAX_W,
    POWER_FAN_ONLY_MAX_W,
    POWER_HEATING_MIN_W,
)

from .model import LinearPowerModel


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


def _get_state(hass: HomeAssistant, entity_id: str) -> State | None:
    return hass.states.get(entity_id)


@dataclass
class ZoneConfig:
    zone_name: str
    climate_entity: str
    room_temp_sensors: list[str]
    power_sensor: str | None = None
    outdoor_temp_sensor: str | None = None
    outdoor_humidity_sensor: str | None = None
    wind_speed_sensor: str | None = None  # for wind chill (optional)


@dataclass
class ZoneRuntime:
    # user controls
    enabled: bool = True
    target_room_temp: float = 20.0
    max_offset_c: float = DEFAULT_MAX_OFFSET_C
    min_offset_c: float = DEFAULT_MIN_OFFSET_C

    # telemetry
    room_temp_c: float | None = None
    climate_setpoint_c: float | None = None
    climate_mode: str | None = None
    fan_mode: str | None = None
    climate_target_temp_step_c: float | None = None
    climate_min_temp_c: float | None = None
    climate_max_temp_c: float | None = None
    actual_power_w: float | None = None
    power_state: str | None = None
    last_power_state: str | None = None
    heating_start_hold_until: Any = None
    predicted_power_w: float | None = None
    last_applied_setpoint_c: float | None = None
    last_setpoint_request_c: float | None = None
    last_setpoint_request_at: Any = None
    desired_setpoint_c: float | None = None
    ramped_setpoint_c: float | None = None
    control_status: str | None = None
    last_update: Any = None

    # learning
    model: LinearPowerModel = field(default_factory=lambda: LinearPowerModel(DEFAULT_K_W_PER_DEG, DEFAULT_BIAS_W))

    # learning telemetry (for UI/graphs)
    learn_updates: int = 0
    last_learn_at: Any = None
    last_learn_offset_c: float | None = None
    last_learn_power_w: float | None = None


class HeatPumpZone:
    def __init__(self, hass: HomeAssistant, entry_id: str, zone_id: str, cfg: ZoneConfig, runtime: ZoneRuntime) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.zone_id = zone_id
        self.cfg = cfg
        self.rt = runtime
        self._unsub = None

    async def async_start(self, interval_seconds: int) -> None:
        async def _tick(_now):
            try:
                await self.async_update_and_control()
            except Exception as err:
                _LOGGER.exception("Zone %s tick failed: %s", self.zone_id, err)

        # Run once immediately so entities get meaningful values quickly.
        self.hass.async_create_task(_tick(dt_util.now()))

        self._unsub = async_track_time_interval(
            self.hass, _tick, timedelta(seconds=interval_seconds)
        )

    async def async_stop(self) -> None:
        if self._unsub:
            self._unsub()
            self._unsub = None

    def _compute_room_temp(self) -> float | None:
        temps: list[float] = []
        for ent in self.cfg.room_temp_sensors:
            st = _get_state(self.hass, ent)
            if not st:
                continue
            v = _safe_float(st.state)
            if v is None:
                continue
            # basic sanity
            if -10.0 <= v <= 45.0:
                temps.append(v)
        if not temps:
            return None
        return sum(temps) / len(temps)

    def _read_power_w(self) -> float | None:
        if not self.cfg.power_sensor:
            return None
        st = _get_state(self.hass, self.cfg.power_sensor)
        if not st:
            return None
        v = _safe_float(st.state)
        if v is None:
            return None
        if 0.0 <= v <= 20000.0:
            return v
        return None

    def _read_outdoor_temp_c(self) -> float | None:
        if not self.cfg.outdoor_temp_sensor:
            return None
        st = _get_state(self.hass, self.cfg.outdoor_temp_sensor)
        if not st:
            return None
        v = _safe_float(st.state)
        if v is None:
            return None
        if -40.0 <= v <= 60.0:
            return v
        return None

    def _read_wind_speed(self) -> float | None:
        if not self.cfg.wind_speed_sensor:
            return None
        st = _get_state(self.hass, self.cfg.wind_speed_sensor)
        if not st:
            return None
        v = _safe_float(st.state)
        if v is None:
            return None
        # Unit varies by integration (m/s, km/h, mph). We only use coarse thresholds.
        if 0.0 <= v <= 80.0:
            return v
        return None

    def _read_climate(self) -> tuple[float | None, str | None, str | None, float | None, float | None, float | None, float | None]:
        st = _get_state(self.hass, self.cfg.climate_entity)
        if not st:
            return None, None, None, None, None, None, None

        attrs = st.attributes or {}
        setpoint = _safe_float(attrs.get("temperature"))
        hvac_mode = attrs.get("hvac_mode") or st.state  # some climates use state
        fan_mode = attrs.get("fan_mode")
        current_temp = _safe_float(attrs.get("current_temperature"))
        target_step = _safe_float(attrs.get("target_temp_step"))
        min_temp = _safe_float(attrs.get("min_temp"))
        max_temp = _safe_float(attrs.get("max_temp"))
        return setpoint, hvac_mode, fan_mode, current_temp, target_step, min_temp, max_temp

    def _clamp_setpoint(self, sp: float) -> float:
        min_temp = self.rt.climate_min_temp_c if self.rt.climate_min_temp_c is not None else DEFAULT_SETPOINT_MIN_C
        max_temp = self.rt.climate_max_temp_c if self.rt.climate_max_temp_c is not None else DEFAULT_SETPOINT_MAX_C
        sp = max(min_temp, min(max_temp, sp))

        # Respect device step size when available; prevents repeated "half-step" requests on 1°C devices.
        step = self.rt.climate_target_temp_step_c
        if step is None or step <= 0 or step > 5:
            # Many heat pumps are whole-degree only; default to 1.0°C when unknown.
            step = 1.0

        # Quantize to step (avoid float noise)
        ticks = round(sp / step)
        sp_q = ticks * step
        return round(sp_q, 3)

    def _ramp(
        self,
        desired: float,
        current: float | None,
        error_c: float,
        power_state: str | None,
        *,
        fast_ramp_error_threshold_c: float = 1.5,
    ) -> float:
        if current is None:
            return desired
        # Respect device step size where possible (many heat pumps are 1.0°C only).
        step = DEFAULT_MAX_SETPOINT_STEP_C_PER_UPDATE
        if self.rt.climate_target_temp_step_c is not None and 0 < self.rt.climate_target_temp_step_c <= 5:
            step = max(step, self.rt.climate_target_temp_step_c)
        # Fast ramp: when the room is well below target and the unit isn't actually heating yet,
        # jump faster to a higher setpoint to start the compressor.
        if error_c >= fast_ramp_error_threshold_c and (power_state is None or power_state in {"idle", "fan_only", "transition"}):
            step = max(step, 2.0)

        if desired > current + step:
            return current + step
        if desired < current - step:
            return current - step
        return desired

    def _choose_offset(self, base_offset: float) -> float:
        """Choose an offset above target.
        """
        offset = max(self.rt.min_offset_c, base_offset)
        return min(self.rt.max_offset_c, max(0.0, offset))

    async def async_apply_setpoint(self, setpoint_c: float) -> None:
        # Anti-beep/anti-spam: don't resend the exact same request repeatedly.
        # Some devices don't reflect the new setpoint immediately, which can cause repeated calls each tick.
        if (
            self.rt.last_setpoint_request_c is not None
            and abs(setpoint_c - self.rt.last_setpoint_request_c) < 1e-3
            and self.rt.last_setpoint_request_at is not None
            and (dt_util.utcnow() - self.rt.last_setpoint_request_at).total_seconds() < 180
        ):
            self.rt.control_status = "cooldown_skip"
            return

        try:
            await self.hass.services.async_call(
                "climate",
                "set_temperature",
                {"entity_id": self.cfg.climate_entity, "temperature": setpoint_c},
                blocking=False,
            )
            self.rt.last_applied_setpoint_c = setpoint_c
            self.rt.last_setpoint_request_c = setpoint_c
            self.rt.last_setpoint_request_at = dt_util.utcnow()
            self.rt.control_status = "applied"
        except Exception as err:
            _LOGGER.error(
                "Failed to set temperature for %s to %.1f°C: %s",
                self.cfg.climate_entity,
                setpoint_c,
                err,
            )
            self.rt.control_status = "apply_failed"

    async def async_update_and_control(self) -> None:
        room_temp = self._compute_room_temp()
        self.rt.room_temp_c = room_temp

        outdoor_temp_c = self._read_outdoor_temp_c()
        wind_speed = self._read_wind_speed()

        setpoint, hvac_mode, fan_mode, _current_temp, target_step, min_temp, max_temp = self._read_climate()
        self.rt.climate_setpoint_c = setpoint
        self.rt.climate_mode = hvac_mode
        self.rt.fan_mode = fan_mode
        self.rt.climate_target_temp_step_c = target_step
        self.rt.climate_min_temp_c = min_temp
        self.rt.climate_max_temp_c = max_temp

        # Predict power from current offset (if room temp known)
        if room_temp is not None and setpoint is not None:
            offset_now = max(0.0, setpoint - room_temp)
            self.rt.predicted_power_w = self.rt.model.predict(offset_now)
        else:
            self.rt.predicted_power_w = None

        # Learning: use actual power + current offset
        power_w = self._read_power_w()
        self.rt.actual_power_w = power_w
        prev_power_state = self.rt.power_state
        if power_w is None:
            self.rt.power_state = None
        elif power_w <= POWER_IDLE_MAX_W:
            self.rt.power_state = "idle"
        elif power_w <= POWER_FAN_ONLY_MAX_W:
            self.rt.power_state = "fan_only"
        elif power_w >= POWER_HEATING_MIN_W:
            self.rt.power_state = "heating"
        else:
            self.rt.power_state = "transition"

        # If we just transitioned into heating, hold further upward ramping briefly.
        # Heat pumps can take a bit to respond; this avoids repeatedly bumping setpoint every 30s.
        if self.rt.power_state == "heating" and prev_power_state in {None, "idle", "fan_only", "transition"}:
            self.rt.heating_start_hold_until = dt_util.utcnow() + timedelta(seconds=DEFAULT_HEATING_START_HOLD_SECONDS)
        self.rt.last_power_state = prev_power_state

        if room_temp is not None and setpoint is not None and power_w is not None:
            offset = max(0.0, setpoint - room_temp)
            # Learn only when we have a meaningful offset and the compressor is actually running.
            if offset >= LEARN_MIN_OFFSET_C and power_w >= LEARN_MIN_POWER_W:
                self.rt.model.update_ema(offset, power_w)
                self.rt.learn_updates += 1
                self.rt.last_learn_at = dt_util.utcnow()
                self.rt.last_learn_offset_c = offset
                self.rt.last_learn_power_w = power_w

        # Control: only if enabled and heating-ish
        if not self.rt.enabled:
            self.rt.control_status = "disabled"
            self.rt.last_update = dt_util.utcnow()
            return

        # If in heat mode or auto heat/cool, we can control setpoint.
        # If off/cool/dry/fan_only, do nothing.
        hvac_mode_s = (hvac_mode or "").lower()
        if hvac_mode_s in {"off", "cool", "dry", "fan_only"}:
            self.rt.control_status = f"hvac_mode_{hvac_mode_s or 'unknown'}"
            self.rt.last_update = dt_util.utcnow()
            return

        if room_temp is None:
            self.rt.control_status = "no_room_temp"
            self.rt.last_update = dt_util.utcnow()
            return

        target = self.rt.target_room_temp
        error = target - room_temp

        if error <= 0:
            # Room is at/above target: drive setpoint back down to target.
            desired_setpoint = self._clamp_setpoint(target)
            self.rt.desired_setpoint_c = desired_setpoint
        else:
            if error < 0.5:
                base_offset = max(self.rt.min_offset_c, 0.4)
            elif error < 1.5:
                base_offset = max(self.rt.min_offset_c, 0.8)
            else:
                base_offset = max(self.rt.min_offset_c, 1.4)

            # Outdoor-aware aggressiveness: colder/windier outside => slightly larger offset.
            # These are small nudges; min/max still apply.
            outdoor_bonus_c = 0.0
            if outdoor_temp_c is not None:
                if outdoor_temp_c <= 0.0:
                    outdoor_bonus_c += 0.2
                if outdoor_temp_c <= -10.0:
                    outdoor_bonus_c += 0.2
                if outdoor_temp_c >= 12.0:
                    outdoor_bonus_c -= 0.1
            if wind_speed is not None:
                if wind_speed >= 5.0:
                    outdoor_bonus_c += 0.1
                if wind_speed >= 10.0:
                    outdoor_bonus_c += 0.1
            outdoor_bonus_c = max(-0.2, min(outdoor_bonus_c, 0.6))

            offset = self._choose_offset(base_offset + outdoor_bonus_c)
            desired_setpoint = self._clamp_setpoint(target + offset)
            self.rt.desired_setpoint_c = desired_setpoint

        # Ramp from current setpoint (or last applied)
        current_sp = setpoint if setpoint is not None else self.rt.last_applied_setpoint_c

        # During the hold window after heating starts, don't increase setpoint further.
        # Allow decreases (e.g., if we overshot or target dropped).
        if (
            current_sp is not None
            and desired_setpoint > current_sp
            and self.rt.heating_start_hold_until is not None
            and dt_util.utcnow() < self.rt.heating_start_hold_until
        ):
            ramped = current_sp
        else:
            # Outdoor-aware ramp: trigger fast-ramp sooner when outside conditions are harsh.
            fast_thresh = 1.5
            if outdoor_temp_c is not None and outdoor_temp_c <= 0.0:
                fast_thresh = 1.0
            if wind_speed is not None and wind_speed >= 10.0:
                fast_thresh = min(fast_thresh, 1.0)

            ramped = self._ramp(
                desired_setpoint,
                current_sp,
                error,
                self.rt.power_state,
                fast_ramp_error_threshold_c=fast_thresh,
            )
        ramped_q = self._clamp_setpoint(ramped)
        self.rt.ramped_setpoint_c = ramped_q

        # Avoid spam: only apply if it changes meaningfully
        if current_sp is None or abs(ramped_q - current_sp) >= 0.25:
            self.rt.control_status = "will_apply"
            await self.async_apply_setpoint(ramped_q)
        else:
            self.rt.control_status = "no_change"

        self.rt.last_update = dt_util.utcnow()
