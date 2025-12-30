# Heat Pump Fabric Manager - Copilot Instructions

## Project Overview
This is a Home Assistant custom integration that manages heat pump climate entities by dynamically adjusting setpoints based on room-temperature error. It uses online learning (EMA) to predict power usage from temperature offsets.

## Architecture

### Core Components
- **HpfmCoordinator** ([coordinator.py](../coordinator.py)): Manages multiple `HeatPumpZone` instances, handles lifecycle (start/stop), and persists learned models via `LearnedStore`
- **HeatPumpZone** ([zone.py](../zone.py)): The heart of the system - each zone runs an independent control loop via `async_track_time_interval`, reading sensors, updating predictions, learning from actual power, and adjusting climate setpoints
- **LinearPowerModel** ([model.py](../model.py)): Simple linear regression `power_w ≈ bias_w + k_w_per_deg * offset_c` updated online using exponential moving average (EMA_ALPHA=0.08)
- **LearnedStore** ([storage.py](../storage.py)): Persists learned model parameters (k, bias) per zone using Home Assistant's Store API

### Data Flow
1. Each zone ticks at configurable intervals (default 30s)
2. Zone reads: room temperature (averaged from multiple sensors), climate setpoint, power sensor, HVAC mode
3. **Learning phase**: If offset ≥ 0.3°C and power ≥ 80W, updates model via EMA
4. **Control phase**: Computes desired offset to reach target room temp, ramps setpoint gradually (max 0.5°C/update), applies via `climate.set_temperature`
5. Model parameters saved every 10 minutes and on shutdown

### Home Assistant Integration Pattern
- Standard config flow setup with options flow (scan interval configurable)
- Creates entities: sensors (room temp, predicted power, learned params, setpoint), numbers (target temp), switches (zone enable/disable)
- Entities use `_attr_has_entity_name = True` convention and share a single device per config entry
- No polling coordinator updates - zones update themselves via time intervals

## Key Design Decisions

### Control Strategy (Option C in `zone.py`)
- **Base offset** adapts to error: small error → gentle offset (0.4-0.8°C), large error → aggressive (1.4°C)
- **Offset selection**: Uses room temperature error to choose an offset, capped at `max_offset_c` (default 3°C)
- **Gentle ramping**: Setpoints change max 0.5°C per update to avoid thermal shock and HVAC wear
- **Anti-spam**: Only applies setpoint changes ≥ 0.25°C difference

### Learning Thresholds
Learning only activates when meaningful signal exists:
- `LEARN_MIN_OFFSET_C = 0.3` (ignore noise near room temp)
- `LEARN_MIN_POWER_W = 80.0` (ignore idle/standby)
- Slow EMA (α=0.08) prioritizes stability over responsiveness

### HVAC Mode Handling
Control only engages when `hvac_mode` is heating-capable. Modes `off`, `cool`, `dry`, `fan_only` are explicitly ignored to prevent interference.

## Code Conventions

### State Access Pattern
Use helper `_get_state(hass, entity_id)` → parse via `_safe_float()` with sanity bounds (temps: -10 to 45°C, power: 0 to 20,000W)

### Dataclass Usage
- `ZoneConfig`: immutable configuration (entity IDs, sensor mappings)
- `ZoneRuntime`: mutable runtime state (enabled, targets, telemetry, learned model)
- Separation enables clean persistence: only `LearnedModel` goes to storage

### Entity Naming
Entities follow `{zone_name} {suffix}` pattern (e.g., "Living Room Predicted Power"). Zone name comes from user input in config flow, zone_id is generated slug.

### Async Lifecycle
- `async_start()`: registers time interval tracker, returns immediately
- `async_stop()`: cancels tracker, awaits cleanup
- `async_persist_learning()`: saves all zone models to storage atomically

## Testing & Development

### Manual Testing Approach
1. Install in Home Assistant custom_components/hp_fabric_manager/
2. Add integration via UI, configure at least one zone with:
   - Climate entity (must support `set_temperature` service)
   - Room temp sensor(s) - multiple sensors are averaged
   - Power sensor (optional but needed for learning)
3. Monitor sensors: watch "Learned K" and "Learned Bias" converge over time
4. Verify setpoint ramping: climate setpoint should change gradually toward target
5. Verify offset clamping: set `max_offset_c` low (e.g., 1°C), observe limited setpoint lift

### Debugging Tips
- Set `scan_interval_seconds` low (10-15s) for faster iteration during development
- Check zone state via developer tools: `hass.data["hp_fabric_manager"][entry_id]["coordinator"].zones[zone_id].rt`
- Learning won't activate if power sensor unavailable or returning None/invalid values
- Setpoint won't update if `enabled=False` or HVAC mode is off/cool

## Common Modifications

### Adding New Control Parameters
1. Add default constant to [const.py](../const.py)
2. Add field to `ZoneRuntime` dataclass in [zone.py](../zone.py)
3. Add config flow schema field in [config_flow.py](../config_flow.py) `async_step_add_zone`
4. Update zone instantiation in [coordinator.py](../coordinator.py) `async_configure_zones`
5. Optionally expose as Number entity in [number.py](../number.py)

### Modifying Learning Algorithm
Edit `LinearPowerModel.update_ema()` in [model.py](../model.py). Current bounds:
- `k_w_per_deg`: clamped to [0, 2000]
- `bias_w`: clamped to [-500, 1500]
Adjust `EMA_ALPHA` in [const.py](../const.py) to change learning speed.

### Changing Control Logic
All control decisions happen in `HeatPumpZone.async_update_and_control()` in [zone.py](../zone.py). The offset calculation is in `_choose_offset()`.
