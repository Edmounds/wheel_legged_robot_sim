#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.controllers.vmc import LEG_CLOSED_LOOP
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import body_id, model_addresses


THETA_MIN = 0.2618
THETA_MAX = 1.6581
THETA_STEP = 0.01
KP = 120.0
KD = 12.0
CTRL_CLIP = 5.0
SETTLE_STEPS = 1500
TAIL = 300


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    model_path = prepare_controlled_mujoco_xml(repo / "sim" / "robot" / "robot.urdf", cache_dir=repo / "tmp" / "probe_full_joint_range")
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
    base_qpos = data.qpos[addr.root_qpos:addr.root_qpos + 7].copy()

    results = []
    for target in np.arange(THETA_MIN, THETA_MAX + 1e-9, THETA_STEP):
        mujoco.mj_resetDataKeyframe(model, data, stand_id)
        mujoco.mj_forward(model, data)
        hs = []
        thetas = []
        for step in range(SETTLE_STEPS):
            for side in LEG_CLOSED_LOOP:
                q = float(data.qpos[motor_qpos[side]])
                qd = float(data.qvel[motor_qvel[side]])
                data.ctrl[motor_acts[side]] = float(np.clip(KP * (target - q) - KD * qd, -CTRL_CLIP, CTRL_CLIP))
            mujoco.mj_step(model, data)
            data.qpos[addr.root_qpos:addr.root_qpos + 7] = base_qpos
            data.qvel[addr.root_qvel:addr.root_qvel + 6] = 0.0
            mujoco.mj_forward(model, data)
            if step >= SETTLE_STEPS - TAIL:
                theta = 0.5 * (float(data.qpos[motor_qpos["left"]]) + float(data.qpos[motor_qpos["right"]]))
                wheel_z = 0.5 * (float(data.xipos[wheel_ids["left"], 2]) + float(data.xipos[wheel_ids["right"], 2]))
                h = float(data.xipos[base_id, 2]) - wheel_z
                thetas.append(theta)
                hs.append(h)
        results.append((float(target), float(np.mean(thetas)), float(np.mean(hs)), float(np.std(hs))))

    minimum = min(results, key=lambda r: r[2])
    maximum = max(results, key=lambda r: r[2])
    print("full joint range cold-start probe")
    print(f"  theta target range=[{THETA_MIN:.4f}, {THETA_MAX:.4f}] step={THETA_STEP:.3f}")
    print(f"  min h: target={minimum[0]:.4f} actual={minimum[1]:.4f} h={minimum[2]:.4f} h_std={minimum[3]:.2e}")
    print(f"  max h: target={maximum[0]:.4f} actual={maximum[1]:.4f} h={maximum[2]:.4f} h_std={maximum[3]:.2e}")
    print("  samples near min:")
    for row in sorted(results, key=lambda r: r[2])[:10]:
        print(f"    target={row[0]:.4f} actual={row[1]:.4f} h={row[2]:.4f} h_std={row[3]:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
