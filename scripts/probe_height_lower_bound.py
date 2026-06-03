#!/usr/bin/env python3
"""One-shot probe: sweep motor angle past current LUT lower bound to find the
true minimum reachable leg height under the same pin-base / zero-gravity
conditions used by ``probe_leg_geometry.py``.

Does NOT overwrite ``leg_height_lut.json``. Prints a table of
(theta_target, theta_actual_mean, h_mean, theta_std, h_std, qvel_max) and
the smallest theta at which the PD probe still converges to its target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.controllers.vmc import LEG_CLOSED_LOOP
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import body_id, model_addresses

# Push down to near the URDF lower limit (0.2618 rad) with a small margin.
THETA_MIN = 0.27
THETA_MAX = 0.65
THETA_STEP = 0.005

KP_PROBE = 80.0
KD_PROBE = 10.0
MOTOR_CTRL_CLIP = 5.0
SETTLE_STEPS = 3000
AVERAGE_WINDOW = 600


def drive(model, data, theta_target, *, motor_acts, motor_qpos, motor_qvel,
          base_id, wheel_ids, root_qpos, root_qvel, base_qpos_pinned):
    theta_L = []
    theta_R = []
    hs = []
    for step in range(SETTLE_STEPS):
        for side in LEG_CLOSED_LOOP:
            theta_cur = float(data.qpos[motor_qpos[side]])
            theta_rate = float(data.qvel[motor_qvel[side]])
            torque = KP_PROBE * (theta_target - theta_cur) - KD_PROBE * theta_rate
            data.ctrl[motor_acts[side]] = float(np.clip(torque, -MOTOR_CTRL_CLIP, MOTOR_CTRL_CLIP))
        mujoco.mj_step(model, data)
        data.qpos[root_qpos:root_qpos + 7] = base_qpos_pinned
        data.qvel[root_qvel:root_qvel + 6] = 0.0
        mujoco.mj_forward(model, data)
        if step >= SETTLE_STEPS - AVERAGE_WINDOW:
            theta_L.append(float(data.qpos[motor_qpos["left"]]))
            theta_R.append(float(data.qpos[motor_qpos["right"]]))
            base_z = float(data.xipos[base_id, 2])
            wl = float(data.xipos[wheel_ids["left"], 2])
            wr = float(data.xipos[wheel_ids["right"], 2])
            hs.append(base_z - 0.5 * (wl + wr))
    theta_mean = 0.5 * (float(np.mean(theta_L)) + float(np.mean(theta_R)))
    h_mean = float(np.mean(hs))
    theta_std = float(np.std(0.5 * (np.array(theta_L) + np.array(theta_R))))
    h_std = float(np.std(hs))
    qvel_max = max(float(abs(data.qvel[motor_qvel[s]])) for s in LEG_CLOSED_LOOP)
    err = theta_target - theta_mean
    return theta_mean, h_mean, theta_std, h_std, qvel_max, err


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    urdf_path = repo_root / "sim" / "robot" / "robot.urdf"
    cache_dir = repo_root / "tmp" / "probe_leg_geometry"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = prepare_controlled_mujoco_xml(urdf_path, cache_dir=cache_dir)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0

    addr = model_addresses(model)
    motor_acts = {s: addr.actuators[g.motor_joint] for s, g in LEG_CLOSED_LOOP.items()}
    motor_qpos = {s: addr.joint_qpos[g.motor_joint] for s, g in LEG_CLOSED_LOOP.items()}
    motor_qvel = {s: addr.joint_qvel[g.motor_joint] for s, g in LEG_CLOSED_LOOP.items()}
    base_id = body_id(model, "base_link")
    wheel_ids = {s: body_id(model, g.wheel_body) for s, g in LEG_CLOSED_LOOP.items()}

    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)
    root_qpos = addr.root_qpos
    root_qvel = addr.root_qvel
    base_qpos_pinned = data.qpos[root_qpos:root_qpos + 7].copy()

    theta_grid = np.arange(THETA_MAX, THETA_MIN - 1e-9, -THETA_STEP)
    print(f"sweep theta from {theta_grid[0]:.4f} down to {theta_grid[-1]:.4f}")
    print(f"URDF joint limit (lower) = 0.2618 rad")
    print(f"HYSTERESIS: each target starts from previous steady state")
    print(f"{'tgt':>6} {'act':>7} {'err':>8} {'h':>8} {'th_std':>9} {'h_std':>9} {'qv_end':>9}")
    print("-" * 64)

    results = []
    for tgt in theta_grid:
        out = drive(model, data, float(tgt),
                    motor_acts=motor_acts, motor_qpos=motor_qpos, motor_qvel=motor_qvel,
                    base_id=base_id, wheel_ids=wheel_ids,
                    root_qpos=root_qpos, root_qvel=root_qvel,
                    base_qpos_pinned=base_qpos_pinned)
        theta_mean, h_mean, th_std, h_std, qvel_max, err = out
        results.append((float(tgt), theta_mean, h_mean, th_std, h_std, qvel_max, err))
        print(f"{tgt:>6.3f} {theta_mean:>7.4f} {err:>+8.4f} {h_mean:>8.4f}"
              f" {th_std:>9.2e} {h_std:>9.2e} {qvel_max:>9.2e}")

    print()
    # Find the smallest target where PD still tracked closely and h was below
    # current LUT min (0.1339).
    converged = [r for r in results if abs(r[6]) < 0.02 and r[5] < 1.0]
    if converged:
        lowest = min(converged, key=lambda r: r[2])
        print(f"Lowest converged: theta_target={lowest[0]:.4f} -> theta_actual={lowest[1]:.4f},"
              f" h={lowest[2]:.4f} (current LUT h_min=0.1339)")
    else:
        print("No targets converged with |err|<0.02 and qvel<1.0 — PD too weak or constraint unstable.")

    # Monotonicity check sorted by theta_actual ascending.
    arr = sorted(results, key=lambda r: r[1])
    hs = [r[2] for r in arr]
    diffs = np.diff(hs)
    sign_changes = int((np.diff(np.sign(diffs)) != 0).sum())
    print(f"h(theta) sign changes across full sweep: {sign_changes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
