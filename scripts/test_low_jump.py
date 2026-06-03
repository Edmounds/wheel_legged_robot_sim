#!/usr/bin/env python3
"""Headless jump from a low STAND posture.

Reproduces the user's reported failure mode: 站立在 cmd_height = h_min (0.078) 时
触发跳跃，腿冲到伸展极限。

We track the motor angle θ over the jump, plus per-phase peaks for h and θ.
If jump_height_min protection works, the motor angle stays inside the LUT's
monotonic-and-reachable range and doesn't bottom out at θ_max.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
from sim.controllers.phase import JumpPhaseMachine, JumpPhaseParams
from sim.controllers.vmc import LEG_CLOSED_LOOP
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.rollout import _clip_control
from sim.state import extract_sim_state, model_addresses


def run(nominal_height: float, duration: float = 4.0, jump_delay: float = 1.0) -> dict:
    urdf_path = Path(__file__).resolve().parent.parent / "sim" / "robot" / "robot.urdf"
    cache_dir = Path(__file__).resolve().parent.parent / "tmp" / "low_jump_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = prepare_controlled_mujoco_xml(urdf_path, cache_dir=cache_dir)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)

    params = deepcopy(STAND_PARAMS)
    params.vmc.nominal_height = nominal_height
    params.vmc.max_height_rate = 100.0

    # NOTE: 此脚本是 jump_height_min 时代的旧产物, 当前 jump pipeline (轨迹规划)
    # 没有这些 phase params 字段. 保留作为 archive — 跑前先修 pm.start_jump 调用.
    phase_params = JumpPhaseParams(
        flight_timeout=0.60,
    )
    pm = JumpPhaseMachine(phase_params)
    controller = CombinedController(params, phase_machine=pm)
    addresses = model_addresses(model)
    motor_q = {side: addresses.joint_qpos[g.motor_joint] for side, g in LEG_CLOSED_LOOP.items()}

    per_phase = {}
    triggered = False
    jumped = False
    steps = int(np.ceil(duration / model.opt.timestep))
    log = []
    last_phase = None
    for i in range(steps):
        state = extract_sim_state(model, data)
        t = i * model.opt.timestep
        if not triggered and t >= jump_delay:
            pm.start_jump()
            triggered = True
        ctrl = controller(model, data, state)
        data.ctrl[:] = _clip_control(model, np.asarray(ctrl, dtype=float))
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            return {"crashed": True, "reason": "nonfinite", "log": log}
        s2 = extract_sim_state(model, data)
        theta_l = float(data.qpos[motor_q["left"]])
        theta_r = float(data.qpos[motor_q["right"]])
        ph = pm.phase.value
        d = per_phase.setdefault(ph, {"theta_max": -10.0, "theta_min": 10.0, "pitch_max": 0.0, "z_max": -10.0, "z_min": 10.0})
        d["theta_max"] = max(d["theta_max"], theta_l, theta_r)
        d["theta_min"] = min(d["theta_min"], theta_l, theta_r)
        d["pitch_max"] = max(d["pitch_max"], abs(float(s2.pitch)))
        d["z_max"] = max(d["z_max"], float(s2.base_position[2]))
        d["z_min"] = min(d["z_min"], float(s2.base_position[2]))
        if ph != last_phase:
            log.append((t, ph, theta_l, theta_r, float(s2.base_position[2]), float(s2.pitch)))
            last_phase = ph
        if abs(float(s2.pitch)) > 1.5:
            return {"crashed": True, "reason": "pitch>1.5", "log": log, "per_phase": per_phase}
    return {"crashed": False, "log": log, "per_phase": per_phase}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", type=float, default=0.078)
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args()
    r = run(args.nominal, args.duration)
    print(f"nominal_height={args.nominal}, crashed={r['crashed']}, reason={r.get('reason')}")
    print(f"LUT range: theta∈[-0.110, 0.650], h∈[0.078, 0.154]")
    print()
    print("phase transitions: t, phase, θ_L, θ_R, base_z, pitch")
    for t, ph, tl, tr, z, pitch in r["log"]:
        print(f"  {t:.3f}  {ph:8s}  θ_L={tl:+.4f}  θ_R={tr:+.4f}  z={z:.4f}  pitch={pitch:+.4f}")
    print()
    print("per-phase peaks (θ in LUT range [-0.110, 0.650]; θ→0.650 == leg extended to limit):")
    print("  phase     θ_min      θ_max      pitch_max  z_min   z_max")
    for ph in ("stand", "crouch", "extend", "flight", "land", "fallen"):
        if ph in r["per_phase"]:
            d = r["per_phase"][ph]
            flag = "  *FULL_EXT*" if d["theta_max"] > 0.62 else ""
            print(f"  {ph:8s}  {d['theta_min']:+.4f}    {d['theta_max']:+.4f}    {d['pitch_max']:.4f}     {d['z_min']:.4f}  {d['z_max']:.4f}{flag}")
    return 0 if not r["crashed"] else 1


if __name__ == "__main__":
    sys.exit(main())
