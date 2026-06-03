#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
from sim.controllers.leg_height_lut import DEFAULT_LUT
from sim.controllers.vmc import _average_leg_height
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import actuator_id, extract_sim_state


WARMUP_SECONDS = 1.0
STEP_SECONDS = 3.0
MIN_CORRELATION = 0.95
MAX_ABS_PITCH = 0.3
MAX_ABS_ROLL = 0.02
MAX_STEP_BASE_XY_DRIFT = 0.08
MAX_STEP_ABS_PITCH = 0.15


def main() -> int:
    params = replace(
        STAND_PARAMS,
        vmc=replace(STAND_PARAMS.vmc),
        target_velocity=0.0,
        target_yaw_rate=0.0,
        fixed_height=False,
    )
    with TemporaryDirectory(dir=Path("tmp")) as tmp_dir:
        model_path = prepare_controlled_mujoco_xml(Path("sim/robot/robot.urdf"), cache_dir=Path(tmp_dir))
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if stand_id < 0:
            raise RuntimeError("missing stand keyframe")
        mujoco.mj_resetDataKeyframe(model, data, stand_id)
        mujoco.mj_forward(model, data)

        height_actuator = actuator_id(model, "cmd_height")
        if height_actuator < 0:
            raise RuntimeError("missing cmd_height actuator")
        command_low = max(float(model.actuator_ctrlrange[height_actuator, 0]), DEFAULT_LUT.h_min)
        command_high = min(float(model.actuator_ctrlrange[height_actuator, 1]), DEFAULT_LUT.h_max)
        nominal_heights = tuple(np.linspace(command_low, command_high, 5).tolist())
        min_pos_z_span = 0.45 * (command_high - command_low)

        controller = CombinedController(params)
        _run_for(model, data, controller, height_actuator, params.vmc.nominal_height, WARMUP_SECONDS)

        rows = []
        max_abs_pitch = 0.0
        max_abs_roll = 0.0
        fell = False
        for nominal_height in nominal_heights:
            params.vmc.nominal_height = nominal_height
            pos_z, leg_h, pitch, roll, base_xy_drift, failed = _run_for(
                model,
                data,
                controller,
                height_actuator,
                nominal_height,
                STEP_SECONDS,
            )
            rows.append((nominal_height, pos_z, leg_h, pitch, roll, base_xy_drift))
            max_abs_pitch = max(max_abs_pitch, abs(pitch))
            max_abs_roll = max(max_abs_roll, abs(roll))
            max_base_xy_drift = max((row[5] for row in rows), default=0.0)
            fell = fell or failed

        commands = np.array([row[0] for row in rows], dtype=float)
        positions = np.array([row[1] for row in rows], dtype=float)
        finite_positions = positions[np.isfinite(positions)]
        if finite_positions.size == positions.size:
            pos_span = float(np.max(positions) - np.min(positions))
            correlation = float(np.corrcoef(commands, positions)[0, 1]) if np.std(positions) > 1e-12 else 0.0
        else:
            pos_span = float("nan")
            correlation = 0.0

        print("cmd_height tracking")
        print(f"  LUT range: [{DEFAULT_LUT.h_min:.4f}, {DEFAULT_LUT.h_max:.4f}] m")
        print(f"  command range: [{command_low:.4f}, {command_high:.4f}] m")
        print(f"  test heights: {[f'{h:.4f}' for h in nominal_heights]}")
        for nominal_height, pos_z, leg_h, pitch, roll, base_xy_drift in rows:
            print(
                f"  nominal={nominal_height:.4f} pos_z={pos_z:.4f} leg_h={leg_h:.4f}"
                f" pitch={pitch:.4f} roll={roll:.4f} base_xy_drift={base_xy_drift:.4f}"
            )
        print(f"  pos_z_span={pos_span:.4f} (threshold {min_pos_z_span:.4f})")
        print(f"  pearson_r={correlation:.4f}")
        print(f"  max_abs_pitch={max_abs_pitch:.4f}")
        print(f"  max_step_abs_pitch_threshold={MAX_STEP_ABS_PITCH:.4f}")
        print(f"  max_abs_roll={max_abs_roll:.4f}")
        print(f"  max_base_xy_drift={max_base_xy_drift:.4f}")
        print(f"  fell={fell}")

        passed = (
            not fell
            and pos_span >= min_pos_z_span
            and correlation > MIN_CORRELATION
            and max_abs_pitch < MAX_ABS_PITCH
            and max_abs_roll < MAX_ABS_ROLL
            and max_abs_pitch < MAX_STEP_ABS_PITCH
            and max_base_xy_drift < MAX_STEP_BASE_XY_DRIFT
        )
        print(f"  result={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1


def _run_for(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: CombinedController,
    height_actuator: int,
    nominal_height: float,
    duration: float,
) -> tuple[float, float, float, float, float, bool]:
    sample_start = max(0, int((duration - 1.0) / model.opt.timestep))
    steps = max(1, int(duration / model.opt.timestep))
    pos_z_samples: list[float] = []
    leg_h_samples: list[float] = []
    pitch_samples: list[float] = []
    roll_samples: list[float] = []
    start_xy: np.ndarray | None = None
    max_xy_drift = 0.0
    failed = False

    for step in range(steps):
        if step == 0:
            start_xy = np.array(extract_sim_state(model, data).base_position[:2], dtype=float)
        state = extract_sim_state(model, data)
        control = np.asarray(controller(model, data, state), dtype=float)
        data.ctrl[:] = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        data.ctrl[height_actuator] = nominal_height
        mujoco.mj_step(model, data)

        state = extract_sim_state(model, data)
        if step >= sample_start:
            pos_z_samples.append(float(state.base_position[2]))
            leg_h_samples.append(_average_leg_height(model, data))
            pitch_samples.append(float(state.pitch))
            roll_samples.append(float(state.roll))
        if start_xy is not None:
            max_xy_drift = max(max_xy_drift, float(np.linalg.norm(state.base_position[:2] - start_xy)))

        if (
            not np.all(np.isfinite(data.qpos))
            or not np.all(np.isfinite(data.qvel))
            or abs(float(state.pitch)) > 1.5
            or float(state.base_position[2]) < 0.02
        ):
            failed = True
            break

    return (
        _mean_or_nan(pos_z_samples),
        _mean_or_nan(leg_h_samples),
        _max_abs_or_nan(pitch_samples),
        _max_abs_or_nan(roll_samples),
        max_xy_drift,
        failed,
    )


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _max_abs_or_nan(values: list[float]) -> float:
    return float(np.max(np.abs(values))) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
