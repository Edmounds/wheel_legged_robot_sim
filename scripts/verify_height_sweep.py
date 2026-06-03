#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.controllers.balance_state import balance_tangent_state_6d
from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
from sim.controllers.leg_height_lut import DEFAULT_LUT
from sim.controllers.vmc import _average_leg_height
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import actuator_id, extract_sim_state


WARMUP_SECONDS = 2.0
DESCEND_SECONDS = 6.0
HOLD_LOW_SECONDS = 2.0
ASCEND_SECONDS = 6.0
TAIL_SECONDS = 1.0

MAX_ABS_PITCH = 0.18
MAX_ABS_ROLL = 0.02
MAX_TAIL_WHEEL_VEL = 0.05


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
        height_low = max(float(model.actuator_ctrlrange[height_actuator, 0]), DEFAULT_LUT.h_min)
        height_high = min(float(model.actuator_ctrlrange[height_actuator, 1]), DEFAULT_LUT.h_max)
        min_pos_z_span = 0.7 * (height_high - height_low)

        controller = CombinedController(params)
        total_seconds = WARMUP_SECONDS + DESCEND_SECONDS + HOLD_LOW_SECONDS + ASCEND_SECONDS
        steps = max(1, int(total_seconds / model.opt.timestep))
        tail_start = max(0, steps - int(TAIL_SECONDS / model.opt.timestep))

        pitch_max = 0.0
        roll_max = 0.0
        descend_pitch_max = 0.0
        low_hold_pitch_max = 0.0
        ascend_pitch_max = 0.0
        base_xy_start: np.ndarray | None = None
        base_xy_max_drift = 0.0
        wheel_pos_start: float | None = None
        wheel_pos_max_drift = 0.0
        wheel_saturation_steps = 0
        leg_saturation_steps = 0
        pos_z_samples: list[float] = []
        leg_h_samples: list[float] = []
        tail_wheel_vel_samples: list[float] = []
        failed = False

        wheel_ranges = model.actuator_ctrlrange[:2]
        leg_ranges = model.actuator_ctrlrange[2:4]

        step = -1
        for step in range(steps):
            t = step * float(model.opt.timestep)
            nominal_height = _height_command(t, height_low, height_high)
            params.vmc.nominal_height = nominal_height

            state = extract_sim_state(model, data)
            control = np.asarray(controller(model, data, state), dtype=float)
            wheel_ctrl = control[:2]
            leg_ctrl = control[2:4]
            if np.any(np.isclose(wheel_ctrl, wheel_ranges[:, 0], atol=1e-6) | np.isclose(wheel_ctrl, wheel_ranges[:, 1], atol=1e-6)):
                wheel_saturation_steps += 1
            if np.any(np.isclose(leg_ctrl, leg_ranges[:, 0], atol=1e-6) | np.isclose(leg_ctrl, leg_ranges[:, 1], atol=1e-6)):
                leg_saturation_steps += 1
            data.ctrl[:] = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[height_actuator] = nominal_height
            mujoco.mj_step(model, data)

            state = extract_sim_state(model, data)
            tangent = balance_tangent_state_6d(model, data, state)
            pitch = abs(float(state.pitch))
            pitch_max = max(pitch_max, pitch)
            phase = _height_phase(t)
            if phase == "descend":
                descend_pitch_max = max(descend_pitch_max, pitch)
            elif phase == "low_hold":
                low_hold_pitch_max = max(low_hold_pitch_max, pitch)
            elif phase == "ascend":
                ascend_pitch_max = max(ascend_pitch_max, pitch)
            roll_max = max(roll_max, abs(float(state.roll)))
            if base_xy_start is None:
                base_xy_start = np.array(state.base_position[:2], dtype=float)
            base_xy_max_drift = max(base_xy_max_drift, float(np.linalg.norm(state.base_position[:2] - base_xy_start)))
            if wheel_pos_start is None:
                wheel_pos_start = float(tangent[4])
            wheel_pos_max_drift = max(wheel_pos_max_drift, abs(float(tangent[4]) - wheel_pos_start))
            pos_z_samples.append(float(state.base_position[2]))
            leg_h_samples.append(_average_leg_height(model, data))
            if step >= tail_start:
                tail_wheel_vel_samples.append(float(tangent[5]))

            if (
                not np.all(np.isfinite(data.qpos))
                or not np.all(np.isfinite(data.qvel))
                or abs(float(state.pitch)) > 1.5
                or float(state.base_position[2]) < 0.02
            ):
                failed = True
                break

        pos_z_span = float(np.max(pos_z_samples) - np.min(pos_z_samples)) if pos_z_samples else float("nan")
        leg_h_min = float(np.min(leg_h_samples)) if leg_h_samples else float("nan")
        leg_h_max = float(np.max(leg_h_samples)) if leg_h_samples else float("nan")
        tail_wheel_vel = float(abs(np.mean(tail_wheel_vel_samples))) if tail_wheel_vel_samples else float("nan")

        total_steps = max(step + 1, 1)
        wheel_saturation_ratio = wheel_saturation_steps / total_steps
        leg_saturation_ratio = leg_saturation_steps / total_steps

        print("height sweep")
        print(f"  LUT range=[{DEFAULT_LUT.h_min:.4f}, {DEFAULT_LUT.h_max:.4f}] m")
        print(f"  command range=[{height_low:.4f}, {height_high:.4f}] m")
        print(f"  leg_h range=[{leg_h_min:.4f}, {leg_h_max:.4f}] m")
        print(f"  pos_z_span={pos_z_span:.4f} (threshold {min_pos_z_span:.4f})")
        print(f"  max_abs_pitch={pitch_max:.4f} (threshold {MAX_ABS_PITCH:.4f})")
        print(f"  descend_max_abs_pitch={descend_pitch_max:.4f}")
        print(f"  low_hold_max_abs_pitch={low_hold_pitch_max:.4f}")
        print(f"  ascend_max_abs_pitch={ascend_pitch_max:.4f}")
        print(f"  max_abs_roll={roll_max:.4f} (threshold {MAX_ABS_ROLL:.4f})")
        print(f"  base_xy_max_drift={base_xy_max_drift:.4f} m")
        print(f"  wheel_pos_max_drift={wheel_pos_max_drift:.4f} m")
        print(f"  wheel_saturation_ratio={wheel_saturation_ratio:.4f}")
        print(f"  leg_saturation_ratio={leg_saturation_ratio:.4f}")
        print(f"  tail_abs_mean_wheel_vel={tail_wheel_vel:.4f} (threshold {MAX_TAIL_WHEEL_VEL:.4f})")
        print(f"  fell={failed}")

        passed = (
            not failed
            and pos_z_span >= min_pos_z_span
            and pitch_max < MAX_ABS_PITCH
            and roll_max < MAX_ABS_ROLL
            and tail_wheel_vel < MAX_TAIL_WHEEL_VEL
        )
        print(f"  result={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1


def _height_command(t: float, height_low: float, height_high: float) -> float:
    if t < WARMUP_SECONDS:
        return height_high
    t -= WARMUP_SECONDS
    if t < DESCEND_SECONDS:
        alpha = t / DESCEND_SECONDS
        return height_high + alpha * (height_low - height_high)
    t -= DESCEND_SECONDS
    if t < HOLD_LOW_SECONDS:
        return height_low
    t -= HOLD_LOW_SECONDS
    alpha = min(t / ASCEND_SECONDS, 1.0)
    return height_low + alpha * (height_high - height_low)


def _height_phase(t: float) -> str:
    if t < WARMUP_SECONDS:
        return "warmup"
    t -= WARMUP_SECONDS
    if t < DESCEND_SECONDS:
        return "descend"
    t -= DESCEND_SECONDS
    if t < HOLD_LOW_SECONDS:
        return "low_hold"
    return "ascend"


if __name__ == "__main__":
    raise SystemExit(main())
