#!/usr/bin/env python3
"""Trace a jump from a low cmd_height to verify LAND target follows nominal."""
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
from sim.state import actuator_id, extract_sim_state


def main() -> int:
    urdf = Path(__file__).resolve().parent.parent / "sim/robot/robot.urdf"
    cache = Path(__file__).resolve().parent.parent / "tmp/low_jump_cache"
    cache.mkdir(parents=True, exist_ok=True)
    model, data = build_controlled_model(urdf, cache_dir=cache, terrain=None)
    controller = create_controlled_controller("lqr_vmc", "stand")
    pm = controller.vmc_controller.phase_machine
    dt = float(model.opt.timestep)
    cmd_jump_id = actuator_id(model, "cmd_jump")

    # Trace at default cmd_height (no lowering — user complained about height even at default)
    for _ in range(int(2.0 / dt)):
        step_controlled_model(model, data, controller)

    print(f"after 2s settle at default cmd_height:")
    print(f"  z={float(data.qpos[2]):.4f}  nominal_height={controller.params.vmc.nominal_height}")

    # Trigger jump
    data.ctrl[cmd_jump_id] = 1.0
    _trigger_jump_on_rising_edge(controller, 1.0, 0.0)
    traj = pm.trajectory
    print(f"jump triggered: h_start={traj.h_start:.4f}  "
          f"h_target_after_land={traj.h_target_after_land:.4f}  "
          f"h_low={traj.h_low:.4f}  h_high={traj.h_high:.4f}")
    print()
    print(f"{'t':>6s} {'phase':>8s} {'z':>7s} {'vz':>7s} {'pitch':>8s} {'ncon':>5s} {'motor_l':>8s}")
    last_phase = "stand"
    for i in range(int(2.0 / dt)):
        state = extract_sim_state(model, data)
        phase = pm.phase.value
        t = i * dt
        if phase != last_phase or i % 20 == 0:
            from sim.state import model_addresses
            addr = model_addresses(model)
            ml = addr.joint_qpos.get("base_link_旋转-2", 7)
            motor = float(data.qpos[ml])
            vz = float(state.base_linear_velocity[2])
            print(f"{t:6.3f} {phase:>8s} {float(data.qpos[2]):7.4f} "
                  f"{vz:+7.3f} {float(state.pitch):+8.3f} {state.contact_count:5d} {motor:+8.3f}")
            last_phase = phase
        if not step_controlled_model(model, data, controller):
            break
        if phase == "stand" and i > int(0.5 / dt):
            print(f"  recovered to STAND at t={t:.3f}, z={float(data.qpos[2]):.4f}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
