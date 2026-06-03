#!/usr/bin/env python3
"""Trace a single jump in detail to diagnose timing."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import mujoco  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.launch_mujoco import (
    build_controlled_model, create_controlled_controller,
    step_controlled_model, _trigger_jump_on_rising_edge,
)
from sim.state import actuator_id, extract_sim_state, model_addresses


def main() -> int:
    urdf = Path(__file__).resolve().parent.parent / "sim/robot/robot.urdf"
    cache = Path(__file__).resolve().parent.parent / "tmp/single_jump_cache"
    cache.mkdir(parents=True, exist_ok=True)
    model, data = build_controlled_model(urdf, cache_dir=cache, terrain=None)
    controller = create_controlled_controller("lqr_vmc", "stand")
    pm = controller.vmc_controller.phase_machine
    dt = float(model.opt.timestep)
    cmd_jump_id = actuator_id(model, "cmd_jump")
    addresses = model_addresses(model)
    wheel_idx = [addresses.joint_qvel[j] for j in addresses.joint_qvel
                 if "link2" in j and ("旋转-13" in j or "旋转-12" in j)]

    # Settle 1s
    for _ in range(int(1.0 / dt)):
        step_controlled_model(model, data, controller)

    print(f"settled: z={float(data.qpos[2]):.4f}")
    # Trigger jump
    data.ctrl[cmd_jump_id] = 1.0
    _trigger_jump_on_rising_edge(controller, 1.0, 0.0)

    print(f"{'t':>6s} {'phase':>8s} {'z':>7s} {'vz':>7s} {'pitch':>8s} {'ncon':>5s} "
          f"{'wlv':>7s} {'wrv':>7s}")
    t = 0.0
    last_phase = "stand"
    for i in range(int(2.0 / dt)):
        state = extract_sim_state(model, data)
        phase = pm.phase.value
        # 起跳前后 (1.295s ~ 1.360s, 即 trace time 0.295-0.360) 每步打印,
        # 其他时间按 10 步 (20ms) 间隔。
        fine_window = 0.295 <= t <= 0.360
        if phase != last_phase or i % 10 == 0 or fine_window:
            wlv = float(data.qvel[wheel_idx[0]])
            wrv = float(data.qvel[wheel_idx[1]])
            vz = float(state.base_linear_velocity[2])
            print(f"{t:6.3f} {phase:>8s} {float(data.qpos[2]):7.4f} "
                  f"{vz:+7.3f} {float(state.pitch):+8.3f} {state.contact_count:5d} "
                  f"{wlv:+7.1f} {wrv:+7.1f}")
            last_phase = phase
        if not step_controlled_model(model, data, controller):
            print(f"{t:6.3f} STEP RETURNED FALSE")
            break
        t += dt
        if phase == "fallen" and i > int(0.05 / dt):
            print(f"{t:6.3f} FALLEN — continuing trace for 100ms more")
            for _ in range(int(0.1 / dt)):
                step_controlled_model(model, data, controller)
                t += dt
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
