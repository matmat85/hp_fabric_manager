from __future__ import annotations

import logging
import math
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
    DEFAULT_NONHEATING_KICK_AFTER_SECONDS,
    DEFAULT_NONHEATING_KICK_INTERVAL_SECONDS,
    DEFAULT_NONHEATING_MIN_ENGAGE_OFFSET_C,
    LEARN_MIN_OFFSET_C,
    LEARN_MIN_POWER_W,
    POWER_IDLE_MAX_W,
    POWER_ANTI_DRAFT_MAX_W,
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
    outdoor_temp_c: float | None = None
    wind_speed: float | None = None
    temp_error_c: float | None = None  # target - room_temp (positive = need heat)
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
    nonheating_below_target_since: Any = None
    last_kick_at: Any = None
    kick_setpoint_c: float | None = None  # the setpoint from the last kick (to hold it)
    kick_debug_waited_s: float | None = None
    kick_debug_since_last_kick_s: float | None = None
    last_action_was_kick: bool = False
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

    @property
    def model_status(self) -> str:
        """Return a status string for the learned model."""
        if self.learn_updates == 0:
            return "untrained"
        elif self.learn_updates < 10:
            return "learning"
        elif self.learn_updates < 50:
            return "warming_up"
        else:
            return "trained"


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

    def _clamp_setpoint_up(self, sp: float) -> float:
        """Clamp like _clamp_setpoint, but quantize upward to the next device step.

        Used for "kick" behavior where rounding down would defeat the purpose.
        """
        min_temp = self.rt.climate_min_temp_c if self.rt.climate_min_temp_c is not None else DEFAULT_SETPOINT_MIN_C
        max_temp = self.rt.climate_max_temp_c if self.rt.climate_max_temp_c is not None else DEFAULT_SETPOINT_MAX_C
        sp = max(min_temp, min(max_temp, sp))

        step = self.rt.climate_target_temp_step_c
        if step is None or step <= 0 or step > 5:
            step = 1.0

        ticks = math.ceil(sp / step)
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
        # Respect device step size where possible.
        # Many heat pumps are whole-degree only; default to 1.0°C when unknown to avoid half-step churn.
        device_step = self.rt.climate_target_temp_step_c
        if device_step is None or device_step <= 0 or device_step > 5:
            device_step = 1.0
        step = max(DEFAULT_MAX_SETPOINT_STEP_C_PER_UPDATE, device_step)
        # Fast ramp: when the room is well below target and the unit isn't actually heating yet,
        # jump faster to a higher setpoint to start the compressor.
        if error_c >= fast_ramp_error_threshold_c and (power_state is None or power_state in {"idle", "anti_draft", "fan_only", "transition"}):
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

        # Store outdoor conditions for visibility
        self.rt.outdoor_temp_c = outdoor_temp_c
        self.rt.wind_speed = wind_speed

        # Calculate and store temperature error
        if room_temp is not None:
            self.rt.temp_error_c = self.rt.target_room_temp - room_temp
        else:
            self.rt.temp_error_c = None

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
        elif power_w <= POWER_ANTI_DRAFT_MAX_W:
            self.rt.power_state = "anti_draft"
        elif power_w <= POWER_FAN_ONLY_MAX_W:
            self.rt.power_state = "fan_only"
        elif power_w >= POWER_HEATING_MIN_W:
            self.rt.power_state = "heating"
        else:
            self.rt.power_state = "transition"

        # If we just transitioned into heating, hold further upward ramping briefly.
        # Heat pumps can take a bit to respond; this avoids repeatedly bumping setpoint every 30s.
        if self.rt.power_state == "heating" and prev_power_state in {None, "idle", "anti_draft", "fan_only", "transition"}:
            self.rt.heating_start_hold_until = dt_util.utcnow() + timedelta(seconds=DEFAULT_HEATING_START_HOLD_SECONDS)
            # Kick succeeded! Clear kick state so we return to normal control.
            if self.rt.kick_setpoint_c is not None:
                self.rt.kick_setpoint_c = None
                self.rt.last_kick_at = None
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

        kicked = False

        if error <= 0:
            # Room is at/above target: drive setpoint back down to target.
            desired_setpoint = self._clamp_setpoint(target)
            self.rt.desired_setpoint_c = desired_setpoint
        else:
            if error < 0.5:
                base_offset = max(self.rt.min_offset_c, 0.4)
            elif error < 1.0:
                base_offset = max(self.rt.min_offset_c, 0.8)
            elif error < 1.5:
                base_offset = max(self.rt.min_offset_c, 1.2)
            else:
                base_offset = max(self.rt.min_offset_c, 1.6)

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

        # Track the "below target but not heating" condition.
        now = dt_util.utcnow()
        nonheating = self.rt.power_state in {None, "idle", "anti_draft", "fan_only", "transition"}
        if error > 0 and nonheating:
            if self.rt.nonheating_below_target_since is None:
                self.rt.nonheating_below_target_since = now
        else:
            self.rt.nonheating_below_target_since = None

        # Ramp from current setpoint (or last applied)
        current_sp = setpoint if setpoint is not None else self.rt.last_applied_setpoint_c

        # Update kick debug info
        if self.rt.nonheating_below_target_since is not None:
            self.rt.kick_debug_waited_s = (now - self.rt.nonheating_below_target_since).total_seconds()
        else:
            self.rt.kick_debug_waited_s = None

        if self.rt.last_kick_at is not None:
            self.rt.kick_debug_since_last_kick_s = (now - self.rt.last_kick_at).total_seconds()
        else:
            self.rt.kick_debug_since_last_kick_s = None

        # Progressive kick logic: if we're below target but heat pump won't start,
        # progressively increase setpoint every KICK_INTERVAL_SECONDS until heating starts.
        # Once heating is confirmed (handled above where power_state becomes "heating"),
        # kick_setpoint_c is cleared and we return to normal control.
        max_allowed_sp = target + float(self.rt.max_offset_c)
        device_step = self.rt.climate_target_temp_step_c
        if device_step is None or device_step <= 0 or device_step > 5:
            device_step = 1.0

        # Determine if we should kick (first kick or progressive kick)
        should_first_kick = (
            error > 0
            and nonheating
            and current_sp is not None
            and self.rt.kick_setpoint_c is None  # not already in kick mode
            and self.rt.nonheating_below_target_since is not None
            and (now - self.rt.nonheating_below_target_since).total_seconds() >= DEFAULT_NONHEATING_KICK_AFTER_SECONDS
        )

        should_progressive_kick = (
            error > 0
            and nonheating
            and current_sp is not None
            and self.rt.kick_setpoint_c is not None  # already in kick mode
            and self.rt.last_kick_at is not None
            and (now - self.rt.last_kick_at).total_seconds() >= DEFAULT_NONHEATING_KICK_INTERVAL_SECONDS
            and self.rt.kick_setpoint_c < max_allowed_sp  # room to increase
        )

        if should_first_kick:
            # First kick: jump to ensure offset above room temp
            kick_target_sp = max(
                desired_setpoint + device_step,
                room_temp + DEFAULT_NONHEATING_MIN_ENGAGE_OFFSET_C,
            )
            kick_target_sp = min(kick_target_sp, max_allowed_sp)
            kick_sp = self._clamp_setpoint_up(kick_target_sp)
            self.rt.kick_setpoint_c = kick_sp
            self.rt.last_kick_at = now
            desired_setpoint = kick_sp
            self.rt.desired_setpoint_c = desired_setpoint
            kicked = True
        elif should_progressive_kick:
            # Progressive kick: bump up by one step
            kick_target_sp = min(self.rt.kick_setpoint_c + device_step, max_allowed_sp)
            kick_sp = self._clamp_setpoint_up(kick_target_sp)
            self.rt.kick_setpoint_c = kick_sp
            self.rt.last_kick_at = now
            desired_setpoint = kick_sp
            self.rt.desired_setpoint_c = desired_setpoint
            kicked = True
        elif self.rt.kick_setpoint_c is not None and nonheating and error > 0:
            # In kick mode but not time for next progressive kick yet - hold the kick setpoint
            desired_setpoint = self.rt.kick_setpoint_c
            self.rt.desired_setpoint_c = desired_setpoint
        else:
            # Not in kick mode, or room reached target, or heating started
            if self.rt.kick_setpoint_c is not None and (not nonheating or error <= 0):
                # Clear kick mode - either heating started or room is satisfied
                self.rt.kick_setpoint_c = None
                self.rt.last_kick_at = None

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

        # Determine control status
        in_kick_hold = self.rt.kick_setpoint_c is not None and not kicked

        # Avoid spam: only apply if it changes meaningfully
        if current_sp is None or abs(ramped_q - current_sp) >= 0.25:
            if kicked:
                self.rt.control_status = "kick_applied"
            elif in_kick_hold:
                self.rt.control_status = "kick_hold"
            else:
                self.rt.control_status = "applied"
            self.rt.last_action_was_kick = kicked
            await self.async_apply_setpoint(ramped_q)
        else:
            if kicked:
                self.rt.control_status = "kick_no_change"
            elif in_kick_hold:
                self.rt.control_status = "kick_hold"
            else:
                self.rt.control_status = "no_change"
            self.rt.last_action_was_kick = False

        self.rt.last_update = dt_util.utcnow()
