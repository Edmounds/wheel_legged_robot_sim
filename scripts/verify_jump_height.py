#!/usr/bin/env python3
"""明确测量单次跳跃高度,精确打印 body z 和 wheel z 峰值。

跑法: uv run python scripts/verify_jump_height.py
"""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.launch_mujoco import (
    build_controlled_model, create_controlled_controller,
    step_controlled_model, _trigger_jump_on_rising_edge,
)
from sim.state import actuator_id, extract_sim_state, body_id


def main() -> int:
    urdf = Path(__file__).resolve().parent.parent / "sim/robot/robot.urdf"
    cache = Path(__file__).resolve().parent.parent / "tmp/single_jump_cache"
    cache.mkdir(parents=True, exist_ok=True)

    # 同 viewer 默认: trapezoid terrain
    model, data = build_controlled_model(
        urdf, cache_dir=cache,
        terrain="single_wheel_trapezoid", terrain_side="left",
    )
    controller = create_controlled_controller("lqr_vmc", "stand")
    dt = float(model.opt.timestep)
    cmd_jump_id = actuator_id(model, "cmd_jump")
    wheel_l_id = body_id(model, "wheel_left")
    wheel_r_id = body_id(model, "wheel_right")

    # 充分 settle (3s)
    for _ in range(int(3.0 / dt)):
        step_controlled_model(model, data, controller)

    z_body_settled = float(data.qpos[2])
    wl_z_settled = float(data.xipos[wheel_l_id, 2])
    wr_z_settled = float(data.xipos[wheel_r_id, 2])
    wheel_z_settled = min(wl_z_settled, wr_z_settled)
    print(f"=== Settled ===")
    print(f"  body z         = {z_body_settled:.4f} m")
    print(f"  wheel z (low)  = {wheel_z_settled:.4f} m")
    print(f"  controller.nominal_height = {controller.params.vmc.nominal_height:.4f}")

    # Trigger with full amplitude
    data.ctrl[cmd_jump_id] = 1.0
    _trigger_jump_on_rising_edge(controller, 1.0, 0.0)
    print(f"\n=== Jump triggered (cmd_jump=1.0) ===")

    body_z_max = z_body_settled
    wheel_z_max = wheel_z_settled
    fallen = False

    for i in range(int(2.0 / dt)):
        state = extract_sim_state(model, data)
        z = float(data.qpos[2])
        wl_z = float(data.xipos[wheel_l_id, 2])
        wr_z = float(data.xipos[wheel_r_id, 2])
        wheel_z = min(wl_z, wr_z)
        body_z_max = max(body_z_max, z)
        wheel_z_max = max(wheel_z_max, wheel_z)
        if abs(float(state.pitch)) > 1.0 or abs(float(state.roll)) > 1.0:
            fallen = True
            break
        if not step_controlled_model(model, data, controller):
            break

    rise_body = (body_z_max - z_body_settled) * 1000
    rise_wheel = (wheel_z_max - wheel_z_settled) * 1000

    print(f"\n=== Peaks ===")
    print(f"  body z peak     = {body_z_max:.4f} m  (rise {rise_body:.1f}mm)")
    print(f"  wheel z peak    = {wheel_z_max:.4f} m  (rise {rise_wheel:.1f}mm)")
    print(f"  fallen          = {fallen}")
    print(f"\n  >= 100mm target: {'PASS' if rise_body >= 100 else 'FAIL'}")
    return 0 if not fallen and rise_body >= 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
