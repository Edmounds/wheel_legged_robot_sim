#!/usr/bin/env python3
"""Headless repeated-jump test compatible with both 227949d (pre-refactor)
and current code. Uses cmd_jump slider only — no JumpTrajectory imports.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.launch_mujoco import (
    build_controlled_model,
    create_controlled_controller,
    step_controlled_model,
    _trigger_jump_on_rising_edge,
)
from sim.state import actuator_id, extract_sim_state, model_addresses


def run(n_jumps: int, interval: float, settle: float) -> int:
    urdf = Path(__file__).resolve().parent.parent / "sim" / "robot" / "robot.urdf"
    cache = Path(__file__).resolve().parent.parent / "tmp" / "repeated_jump_cache"
    cache.mkdir(parents=True, exist_ok=True)
    model, data = build_controlled_model(urdf, cache_dir=cache, terrain=None)
    controller = create_controlled_controller("lqr_vmc", "stand")
    pm = getattr(getattr(controller, "vmc_controller", controller), "phase_machine", None)
    if pm is None:
        raise RuntimeError("phase_machine not found on controller")

    addresses = model_addresses(model)
    wheel_qvel_idx = [
        addresses.joint_qvel[j] for j in addresses.joint_qvel
        if "link2" in j and ("旋转-13" in j or "旋转-12" in j)
    ]
    if len(wheel_qvel_idx) != 2:
        raise RuntimeError(f"could not locate 2 wheel qvels: {wheel_qvel_idx}")

    dt = float(model.opt.timestep)
    initial_z = float(data.qpos[2])
    cmd_jump_id = actuator_id(model, "cmd_jump")

    for _ in range(int(settle / dt)):
        assert step_controlled_model(model, data, controller)

    print(f"After {settle:.1f}s pre-roll: z={float(data.qpos[2]):.4f} "
          f"pitch={float(extract_sim_state(model, data).pitch):+.4f} "
          f"wheel_pos=({float(data.qpos[-2]):+.3f},{float(data.qpos[-1]):+.3f})")

    prev_cmd = 0.0
    clean = 0
    for jump_idx in range(n_jumps):
        # Rising-edge trigger
        data.ctrl[cmd_jump_id] = 1.0
        prev_cmd = _trigger_jump_on_rising_edge(controller, 1.0, prev_cmd)

        m = {"idx": jump_idx + 1, "z_peak": -10.0, "z_min": 10.0,
             "pitch_max": 0.0, "wheel_v_max": 0.0,
             "vx_max": 0.0, "vy_max": 0.0, "phases": [pm.phase.value],
             "airborne_ms": 0.0, "fallen": False}

        for _ in range(int(interval / dt)):
            state = extract_sim_state(model, data)
            m["z_peak"] = max(m["z_peak"], float(state.base_position[2]))
            m["z_min"] = min(m["z_min"], float(state.base_position[2]))
            m["pitch_max"] = max(m["pitch_max"], abs(float(state.pitch)))
            wv = max(abs(float(data.qvel[wheel_qvel_idx[0]])),
                     abs(float(data.qvel[wheel_qvel_idx[1]])))
            m["wheel_v_max"] = max(m["wheel_v_max"], wv)
            m["vx_max"] = max(m["vx_max"], abs(float(state.base_linear_velocity[0])))
            m["vy_max"] = max(m["vy_max"], abs(float(state.base_linear_velocity[1])))
            if state.contact_count == 0:
                m["airborne_ms"] += dt * 1000.0
            if pm.phase.value != m["phases"][-1]:
                m["phases"].append(pm.phase.value)
            if pm.phase.value == "fallen":
                m["fallen"] = True
            if not step_controlled_model(model, data, controller):
                m["fallen"] = True
                break
            # Slider stays at 1.0 between iterations — that's the prod behavior
            # when user holds the slider up
        # Drop the slider so next iteration can rising-edge
        data.ctrl[cmd_jump_id] = 0.0
        prev_cmd = _trigger_jump_on_rising_edge(controller, 0.0, prev_cmd)
        # Let it settle a bit before next attempt
        for _ in range(int(0.3 / dt)):
            assert step_controlled_model(model, data, controller)

        final_state = extract_sim_state(model, data)
        m["final_z"] = float(data.qpos[2])
        m["final_pitch"] = float(final_state.pitch)
        m["final_phase"] = pm.phase.value
        m["final_wheel_pos"] = (float(data.qpos[-2]), float(data.qpos[-1]))
        ok = (not m["fallen"] and m["final_phase"] == "stand"
              and abs(m["final_pitch"]) < 0.2)
        if ok:
            clean += 1
        status = "[OK]" if ok else "[FAIL]"
        print(f"\n#{m['idx']} {status} phases: {' → '.join(m['phases'])}")
        print(f"   lift={m['z_peak']-initial_z:+.3f}m  pitch_max={m['pitch_max']:.2f}  "
              f"wheel_v_max={m['wheel_v_max']:.0f}rad/s  airborne={m['airborne_ms']:.0f}ms")
        print(f"   drift: vx_max={m['vx_max']:.2f}  vy_max={m['vy_max']:.2f}m/s")
        print(f"   final: z={m['final_z']:.3f}  pitch={m['final_pitch']:+.3f}  "
              f"wheel_pos=({m['final_wheel_pos'][0]:+.2f},{m['final_wheel_pos'][1]:+.2f})  "
              f"phase={m['final_phase']}")
        if m["fallen"]:
            print("   stopping — FALLEN")
            break

    print(f"\n=== {clean}/{n_jumps} clean ===")
    return 0 if clean == n_jumps else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jumps", type=int, default=5)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--settle", type=float, default=2.0)
    args = ap.parse_args()
    return run(args.jumps, args.interval, args.settle)


if __name__ == "__main__":
    sys.exit(main())
