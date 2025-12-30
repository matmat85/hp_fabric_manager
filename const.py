from __future__ import annotations

DOMAIN = "hp_fabric_manager"
PLATFORMS: list[str] = ["sensor", "number", "switch"]

DEFAULT_SCAN_INTERVAL_SECONDS = 30

# Control behaviour defaults
DEFAULT_TARGET_ROOM_TEMP = 20.0
DEFAULT_MAX_OFFSET_C = 3.0
DEFAULT_MIN_OFFSET_C = 0.2
DEFAULT_MAX_SETPOINT_STEP_C_PER_UPDATE = 0.5  # gentle ramp
DEFAULT_SETPOINT_MIN_C = 16.0
DEFAULT_SETPOINT_MAX_C = 32.0
DEFAULT_HEATING_START_HOLD_SECONDS = 120

# If the room is below target but the heat pump appears "satisfied" (fan-only/idle),
# we may need to temporarily push setpoint higher to re-engage heating.
DEFAULT_NONHEATING_KICK_AFTER_SECONDS = 60
DEFAULT_NONHEATING_KICK_STEP_C = 1.0
DEFAULT_NONHEATING_KICK_COOLDOWN_SECONDS = 600
DEFAULT_NONHEATING_MIN_ENGAGE_OFFSET_C = 2.0

# Learning defaults
DEFAULT_K_W_PER_DEG = 200.0  # initial guess: +1C offset -> +200W
DEFAULT_BIAS_W = 0.0         # baseline
EMA_ALPHA = 0.08             # how fast we learn (0..1). small = slow & stable

# Power-state heuristics (based on typical heat pump behavior)
# - ~2-5W: idle/standby
# - ~10-65W: fan-only / satisfied circulation
# - >=100W: compressor/heating is actually running
POWER_IDLE_MAX_W = 5.0
POWER_FAN_ONLY_MAX_W = 65.0
POWER_HEATING_MIN_W = 100.0

# When to learn: ignore tiny offsets / near-idle noise
LEARN_MIN_OFFSET_C = 0.3
LEARN_MIN_POWER_W = POWER_HEATING_MIN_W

ATTR_ZONE = "zone"
