from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
from sim.controllers.vmc import LEG_CLOSED_LOOP
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import actuator_id, body_id, extract_sim_state, model_addresses

TARGET_RAMP_SURFACE_HEIGHT = 0.012
SETTLE_DURATION = 1.0
RAMP_ENTRY_MARGIN = 0.012


@dataclass(frozen=True)
class RampMetrics:
    steps: int
    finite: bool
    fell: bool
    reached_ramp_target: bool
    failure_reason: str | None
    initial_base_y: float
    initial_ramp_wheel_y: float
    initial_ground_wheel_y: float
    forward_distance: float
    ramp_start_y: float
    ramp_target_y: float
    target_ramp_surface_height: float
    ramp_wheel_lift: float
    ground_wheel_lift: float
    final_wheel_height_delta: float
    max_abs_pitch: float
    max_abs_roll: float
    final_abs_roll: float
    tail_max_abs_roll: float
    max_base_z_delta: float
    max_wheel_height_delta: float
    contact_count: int
    saturation_ratio: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a headless single-wheel trapezoid ramp VMC stability check.")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--target-velocity", type=float, default=0.12)
    parser.add_argument("--terrain-side", choices=("left", "right"), default="left")
    parser.add_argument("--enable-roll-leveling", action="store_true")
    parser.add_argument(
        "--full-platform",
        action="store_true",
        help="drive the ramp wheel fully onto the platform (max left/right wheel-height delta)",
    )
    args = parser.parse_args()

    tmp_root = Path("tmp")
    tmp_root.mkdir(exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="single_wheel_ramp_", dir=tmp_root))
    try:
        metrics = run_single_wheel_ramp(
            duration=args.duration,
            target_velocity=args.target_velocity,
            terrain_side=args.terrain_side,
            enable_roll_leveling=args.enable_roll_leveling,
            full_platform=args.full_platform,
            cache_dir=temp_dir,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"steps={metrics.steps}")
    print(f"finite={metrics.finite}")
    print(f"fell={metrics.fell}")
    print(f"reached_ramp_target={metrics.reached_ramp_target}")
    print(f"failure_reason={metrics.failure_reason}")
    print(f"initial_base_y={metrics.initial_base_y:.4f} m")
    print(f"initial_ramp_wheel_y={metrics.initial_ramp_wheel_y:.4f} m")
    print(f"initial_ground_wheel_y={metrics.initial_ground_wheel_y:.4f} m")
    print(f"forward_distance={metrics.forward_distance:.4f} m")
    print(f"ramp_start_y={metrics.ramp_start_y:.4f} m")
    print(f"ramp_target_y={metrics.ramp_target_y:.4f} m")
    print(f"target_ramp_surface_height={metrics.target_ramp_surface_height:.4f} m")
    print(f"ramp_wheel_lift={metrics.ramp_wheel_lift:.4f} m")
    print(f"ground_wheel_lift={metrics.ground_wheel_lift:.4f} m")
    print(f"final_wheel_height_delta={metrics.final_wheel_height_delta:.4f} m")
    print(f"max_abs_pitch={metrics.max_abs_pitch:.4f} rad")
    print(f"max_abs_roll={metrics.max_abs_roll:.4f} rad")
    print(f"final_abs_roll={metrics.final_abs_roll:.4f} rad")
    print(f"tail_max_abs_roll={metrics.tail_max_abs_roll:.4f} rad")
    print(f"max_base_z_delta={metrics.max_base_z_delta:.4f} m")
    print(f"max_wheel_height_delta={metrics.max_wheel_height_delta:.4f} m")
    print(f"contact_count={metrics.contact_count}")
    print(f"saturation_ratio={metrics.saturation_ratio:.4f}")
    if args.full_platform:
        # 单轮全程爬上 65mm 平台是接近腿差极限 (~±37.6mm) 的工况, 阈值比 12mm 浅坡放宽。
        roll_ok = metrics.final_abs_roll <= 0.05 and metrics.tail_max_abs_roll <= 0.08
        sat_ok = metrics.saturation_ratio <= 0.35
    else:
        roll_ok = metrics.final_abs_roll <= 0.02 and metrics.tail_max_abs_roll <= 0.04
        sat_ok = metrics.saturation_ratio <= 0.20
    passed = (
        metrics.finite
        and not metrics.fell
        and metrics.reached_ramp_target
        and roll_ok
        and sat_ok
    )
    return 0 if passed else 1


def run_single_wheel_ramp(
    *,
    duration: float,
    target_velocity: float,
    terrain_side: str,
    cache_dir: Path,
    enable_roll_leveling: bool = False,
    full_platform: bool = False,
) -> RampMetrics:
    model_path = prepare_controlled_mujoco_xml(
        Path("sim/robot/robot.urdf"),
        cache_dir=cache_dir,
        terrain="single_wheel_trapezoid",
        terrain_side=terrain_side,
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)

    ramp_target = _ramp_target(model, data, terrain_side, full_platform)
    addresses = model_addresses(model)
    preplace_ramp_wheel_y = _wheel_y(model, data, ramp_target.ramp_side)
    spawn_delta_y = float(ramp_target.ramp_start_y - RAMP_ENTRY_MARGIN - preplace_ramp_wheel_y)
    data.qpos[addresses.root_qpos + 1] += spawn_delta_y
    mujoco.mj_forward(model, data)

    params = replace(
        STAND_PARAMS,
        vmc=replace(STAND_PARAMS.vmc),
        q_diag=STAND_PARAMS.q_diag.copy(),
        r_diag=STAND_PARAMS.r_diag.copy(),
        target_velocity=float(target_velocity),
    )
    if not enable_roll_leveling:
        params.vmc.roll_level_kp_height = 0.0
        params.vmc.roll_level_kd_height = 0.0
        params.vmc.roll_level_offset_limit = 0.0
        params.vmc.slope_squat_margin = 0.0
    # enabled: keep STAND_PARAMS production defaults (feedforward leveling + slope squat).
    controller = CombinedController(params)
    controller.params.target_velocity = float(target_velocity)
    linear_cmd_act = actuator_id(model, "cmd_linear_x")
    height_cmd_act = actuator_id(model, "cmd_height")

    initial_state = extract_sim_state(model, data)
    initial_forward_y = float(initial_state.base_position[1])
    initial_z = float(initial_state.base_position[2])
    ramp_wheel_initial_z = _wheel_z(model, data, ramp_target.ramp_side)
    ground_wheel_initial_z = _wheel_z(model, data, ramp_target.ground_side)
    ramp_wheel_initial_y = _wheel_y(model, data, ramp_target.ramp_side)
    ground_wheel_initial_y = _wheel_y(model, data, ramp_target.ground_side)
    max_abs_pitch = abs(float(initial_state.pitch))
    max_abs_roll = abs(float(initial_state.roll))
    tail_max_abs_roll = 0.0
    max_base_z_delta = 0.0
    max_wheel_height_delta = _wheel_height_delta(model, data)
    contact_count = int(data.ncon)
    saturated_controls = 0
    total_controls = 0
    finite = True
    failure_reason: str | None = None
    executed_steps = 0
    reached_ramp_target = False
    settle_steps = 0
    ramp_wheel_lift = 0.0
    ground_wheel_lift = 0.0

    step_count = max(1, int(np.ceil(float(duration) / float(model.opt.timestep))))
    for _ in range(step_count):
        state = extract_sim_state(model, data)
        if reached_ramp_target:
            controller.params.target_velocity = 0.0
        else:
            controller.params.target_velocity = float(target_velocity)
        control = np.asarray(controller(model, data, state), dtype=float)
        if control.shape != (model.nu,) or not np.all(np.isfinite(control)):
            finite = False
            failure_reason = "invalid_control"
            break

        clipped = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        saturated_controls += int(np.count_nonzero(np.abs(clipped - control) > 1e-12))
        total_controls += int(model.nu)
        data.ctrl[:] = clipped
        if linear_cmd_act >= 0:
            data.ctrl[linear_cmd_act] = float(controller.params.target_velocity)
        if height_cmd_act >= 0:
            data.ctrl[height_cmd_act] = float(controller.params.vmc.nominal_height)
        mujoco.mj_step(model, data)
        executed_steps += 1

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            finite = False
            failure_reason = "nonfinite_state"
            break

        state = extract_sim_state(model, data)
        elapsed = executed_steps * float(model.opt.timestep)
        ramp_wheel_y = _wheel_y(model, data, ramp_target.ramp_side)
        ramp_wheel_lift_now = _wheel_z(model, data, ramp_target.ramp_side) - ramp_wheel_initial_z
        if not reached_ramp_target:
            if full_platform:
                # 全平台工况按"轮已抬到平台高度"判定到位 (与前进速度解耦): 找平会
                # 减慢前进, 用 y 位置判定会冤枉慢但调平成功的运行。
                reached_ramp_target = ramp_wheel_lift_now >= 0.9 * ramp_target.surface_height
            elif ramp_wheel_y >= ramp_target.target_y:
                reached_ramp_target = True
        if reached_ramp_target:
            settle_steps += 1
        max_abs_pitch = max(max_abs_pitch, abs(float(state.pitch)))
        max_abs_roll = max(max_abs_roll, abs(float(state.roll)))
        if reached_ramp_target:
            tail_max_abs_roll = max(tail_max_abs_roll, abs(float(state.roll)))
        max_base_z_delta = max(max_base_z_delta, abs(float(state.base_position[2]) - initial_z))
        max_wheel_height_delta = max(max_wheel_height_delta, _wheel_height_delta(model, data))
        ramp_wheel_lift = max(ramp_wheel_lift, _wheel_z(model, data, ramp_target.ramp_side) - ramp_wheel_initial_z)
        ground_wheel_lift = max(ground_wheel_lift, abs(_wheel_z(model, data, ramp_target.ground_side) - ground_wheel_initial_z))
        contact_count = max(contact_count, int(state.contact_count))
        if elapsed > 1.0 and state.base_position[2] < initial_z - 0.15:
            failure_reason = "fell"
            break
        if abs(float(state.pitch)) > 1.5 or abs(float(state.roll)) > 1.5:
            failure_reason = "excessive_attitude"
            break
        if reached_ramp_target and settle_steps * float(model.opt.timestep) >= SETTLE_DURATION:
            break

    if finite and failure_reason is None and not reached_ramp_target:
        failure_reason = "ramp_target_not_reached"

    final_state = extract_sim_state(model, data)
    final_wheel_height_delta = _wheel_height_delta(model, data)
    return RampMetrics(
        steps=executed_steps,
        finite=finite,
        fell=failure_reason in {"fell", "excessive_attitude"},
        reached_ramp_target=reached_ramp_target,
        failure_reason=failure_reason,
        initial_base_y=initial_forward_y,
        initial_ramp_wheel_y=ramp_wheel_initial_y,
        initial_ground_wheel_y=ground_wheel_initial_y,
        forward_distance=float(final_state.base_position[1] - initial_forward_y),
        ramp_start_y=ramp_target.ramp_start_y,
        ramp_target_y=ramp_target.target_y,
        target_ramp_surface_height=ramp_target.surface_height,
        ramp_wheel_lift=ramp_wheel_lift,
        ground_wheel_lift=ground_wheel_lift,
        final_wheel_height_delta=final_wheel_height_delta,
        max_abs_pitch=max_abs_pitch,
        max_abs_roll=max_abs_roll,
        final_abs_roll=abs(float(final_state.roll)),
        tail_max_abs_roll=tail_max_abs_roll,
        max_base_z_delta=max_base_z_delta,
        max_wheel_height_delta=max_wheel_height_delta,
        contact_count=contact_count,
        saturation_ratio=(saturated_controls / total_controls) if total_controls else 0.0,
    )


@dataclass(frozen=True)
class RampTarget:
    ramp_side: str
    ground_side: str
    ramp_start_y: float
    ramp_end_y: float
    target_y: float
    surface_height: float


def _ramp_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    terrain_side: str,
    full_platform: bool = False,
) -> RampTarget:
    ramp_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "single_wheel_trapezoid_ramp_up")
    platform_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "single_wheel_trapezoid_platform")
    if ramp_geom < 0 or platform_geom < 0:
        raise ValueError("single-wheel ramp terrain is missing")

    platform_height = float(data.geom_xpos[platform_geom, 2] + model.geom_size[platform_geom, 2])
    ramp_span = float(2.0 * model.geom_size[ramp_geom, 1])
    ramp_length = float(np.sqrt(max(ramp_span * ramp_span - platform_height * platform_height, 0.0)))
    ramp_start_y = float(data.geom_xpos[platform_geom, 1] - model.geom_size[platform_geom, 1] - ramp_length)
    ramp_end_y = float(data.geom_xpos[platform_geom, 1] + model.geom_size[platform_geom, 1] + ramp_length)
    if full_platform:
        # 驱动到平台中点: 单轮完全爬上 platform_height 平台, 制造满量程左右轮高差。
        surface_height = platform_height
        target_y = float(data.geom_xpos[platform_geom, 1])
    else:
        surface_height = min(TARGET_RAMP_SURFACE_HEIGHT, 0.75 * platform_height)
        target_y = ramp_start_y + ramp_length * surface_height / platform_height
    return RampTarget(
        ramp_side=terrain_side,
        ground_side="right" if terrain_side == "left" else "left",
        ramp_start_y=ramp_start_y,
        ramp_end_y=ramp_end_y,
        target_y=target_y,
        surface_height=surface_height,
    )


def _wheel_y(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    return float(data.xipos[body_id(model, LEG_CLOSED_LOOP[side].wheel_body), 1])


def _wheel_z(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    return float(data.xipos[body_id(model, LEG_CLOSED_LOOP[side].wheel_body), 2])


def _wheel_height_delta(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    heights = [
        float(data.xipos[body_id(model, geometry.wheel_body), 2])
        for geometry in LEG_CLOSED_LOOP.values()
    ]
    return abs(heights[0] - heights[1])


if __name__ == "__main__":
    raise SystemExit(main())
