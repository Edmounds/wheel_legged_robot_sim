from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

mujoco = cast(Any, importlib.import_module("mujoco"))
import numpy as np

from src.metrics import RolloutMetrics
from src.mjcf_builder import prepare_controlled_mujoco_xml
from src.state import SimState, extract_sim_state, model_addresses


Controller = Callable[[Any, Any, SimState], np.ndarray]


@dataclass(frozen=True)
class RolloutConfig:
    duration: float
    scenario: str = "stand"
    cache_dir: Path | None = None
    fall_height_drop: float = 0.15  # 相对初始高度的下降量阈值


@dataclass(frozen=True)
class RolloutResult:
    model: Any
    data: Any
    metrics: RolloutMetrics
    steps: int
    initial_base_position: np.ndarray
    last_control: np.ndarray
    failure_reason: str | None = None


def zero_controller(model: Any, data: Any, state: SimState) -> np.ndarray:
    del data, state
    return np.zeros(model.nu)


def run_rollout(xml_path: Path, config: RolloutConfig, controller: Controller = zero_controller) -> RolloutResult:
    model_path = prepare_controlled_mujoco_xml(xml_path, cache_dir=config.cache_dir)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)
    _apply_rollout_scenario_initial_state(model, data, config.scenario)

    initial_state = extract_sim_state(model, data)
    initial_base_position = initial_state.base_position.copy()
    min_base_height = float(initial_base_position[2])
    max_base_height = float(initial_base_position[2])
    max_abs_pitch = abs(float(initial_state.pitch))
    max_abs_y_drift = 0.0
    contact_count = int(data.ncon)
    control_effort = 0.0
    saturated_controls = 0
    total_controls = 0
    last_control = np.zeros(model.nu)
    finite = True
    failure_reason: str | None = None
    executed_steps = 0
    
    is_driving_phase = config.scenario == "drive"
    drive_start_step = 0
    drive_start_dist = float(initial_base_position[0])

    step_count = max(1, int(np.ceil(config.duration / model.opt.timestep)))
    for _ in range(step_count):
        state = extract_sim_state(model, data)
        time_elapsed = executed_steps * model.opt.timestep
        
        # stand_then_drive phase switching
        if config.scenario == "stand_then_drive":
            if not is_driving_phase:
                params = getattr(controller, "params", None)
                if params is not None:
                    params.target_velocity = 0.0
                
                # Fast-forward check: if completely still after 10s (steady state) or reached limit
                if (time_elapsed > 10.0 and abs(float(state.pitch_rate)) < 0.05 and abs(float(state.base_linear_velocity[1])) < 0.05) or time_elapsed >= 600.0:
                    is_driving_phase = True
                    drive_start_step = executed_steps
                    drive_start_dist = float(state.base_position[0])
            
            if is_driving_phase:
                params = getattr(controller, "params", None)
                if params is not None:
                    drive_time = time_elapsed - (drive_start_step * float(model.opt.timestep))
                    # 3 秒内平滑加速到 0.45 m/s。
                    params.target_velocity = min(0.45, 0.45 * drive_time / 3.0)

        control = np.asarray(controller(model, data, state), dtype=float)
        if control.shape != (model.nu,):
            finite = False
            failure_reason = "invalid control shape"
            break
        if not np.all(np.isfinite(control)):
            finite = False
            failure_reason = "nonfinite control"
            last_control = control
            break

        clipped_control = _clip_control(model, control)
        control_effort += float(np.sum(np.square(clipped_control)))
        saturated_controls += int(np.count_nonzero(np.abs(clipped_control - control) > 1e-12))
        total_controls += model.nu
        data.ctrl[:] = clipped_control
        last_control = clipped_control.copy()
        mujoco.mj_step(model, data)
        executed_steps += 1

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            finite = False
            failure_reason = "nonfinite state"
            break

        state = extract_sim_state(model, data)
        # 启动瞬态免疫期 (1s) 后才跟踪 min_base_height
        if time_elapsed > 1.0:
            min_base_height = min(min_base_height, float(state.base_position[2]))
        max_base_height = max(max_base_height, float(state.base_position[2]))
        max_abs_pitch = max(max_abs_pitch, abs(float(state.pitch)))
        max_abs_y_drift = max(max_abs_y_drift, abs(float(state.base_position[1] - initial_base_position[1])))
        contact_count = max(contact_count, int(state.contact_count))

        # Early termination: Fall detection (前 1s 为启动瞬态免疫期)
        if (
            config.scenario != "fall_recover"
            and time_elapsed > 1.0
            and state.base_position[2] < initial_base_position[2] - config.fall_height_drop
        ):
            failure_reason = "fell"
            break

        # Early termination: Excessive pitch (loss of balance)
        if config.scenario != "fall_recover" and abs(float(state.pitch)) > 1.5:
            failure_reason = "excessive_pitch"
            break

        # Check drive completion
        current_dist = float(state.base_position[0] - initial_base_position[0])
        if config.scenario == "stand_then_drive" and is_driving_phase and current_dist >= 10.0:
            break

        # Early termination: Stagnation for drive scenario (or drive phase of stand_then_drive)
        if is_driving_phase:
            drive_elapsed_steps = executed_steps - drive_start_step
            if drive_elapsed_steps == int(2.0 / model.opt.timestep):
                # Using 2 seconds to check stagnation to be generous
                drive_progress = float(state.base_position[0] - drive_start_dist)
                if drive_progress < 0.1:
                    failure_reason = "stagnation"
                    break

    final_state = extract_sim_state(model, data)
    saturation_ratio = saturated_controls / total_controls if total_controls else 0.0
    is_fell = failure_reason in ("fell", "excessive_pitch", "stagnation") or bool(min_base_height < initial_base_position[2] - config.fall_height_drop)
    
    metrics = RolloutMetrics(
        duration=config.duration,
        finite=finite,
        fell=is_fell,
        min_base_height=min_base_height,
        max_base_height=max_base_height,
        max_jump_height=max(0.0, max_base_height - float(initial_base_position[2])),
        max_abs_pitch=max_abs_pitch,
        max_abs_y_drift=max_abs_y_drift,
        forward_distance=float(final_state.base_position[0] - initial_base_position[0]),
        contact_count=contact_count,
        control_effort=control_effort,
        saturation_ratio=saturation_ratio,
    )
    return RolloutResult(
        model=model,
        data=data,
        metrics=metrics,
        steps=executed_steps,
        initial_base_position=initial_base_position,
        last_control=last_control,
        failure_reason=failure_reason,
    )


def _apply_rollout_scenario_initial_state(model: Any, data: Any, scenario: str) -> None:
    if scenario != "fall_recover":
        return
    addresses = model_addresses(model)
    root_qpos = addresses.root_qpos
    pitch_angle = 1.2
    data.qpos[root_qpos + 3 : root_qpos + 7] = np.array(
        [np.cos(-pitch_angle / 2.0), np.sin(-pitch_angle / 2.0), 0.0, 0.0],
        dtype=float,
    )
    data.qvel[addresses.root_qvel : addresses.root_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)


def _clip_control(model: Any, control: np.ndarray) -> np.ndarray:
    if model.nu == 0:
        return control
    return np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
