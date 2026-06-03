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
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import actuator_id, extract_sim_state


DURATION_SECONDS = 10.0
TAIL_SECONDS = 1.0
MAX_TAIL_WHEEL_VEL = 0.02
MAX_TAIL_WHEEL_POS_DRIFT = 0.05


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

        controller = CombinedController(params)
        steps = max(1, int(DURATION_SECONDS / model.opt.timestep))
        tail_start = max(0, steps - int(TAIL_SECONDS / model.opt.timestep))
        wheel_pos_samples: list[float] = []
        wheel_vel_samples: list[float] = []
        failed = False

        for step in range(steps):
            state = extract_sim_state(model, data)
            control = np.asarray(controller(model, data, state), dtype=float)
            data.ctrl[:] = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[height_actuator] = params.vmc.nominal_height
            mujoco.mj_step(model, data)

            state = extract_sim_state(model, data)
            tangent = balance_tangent_state_6d(model, data, state)
            if step >= tail_start:
                wheel_pos_samples.append(float(tangent[4]))
                wheel_vel_samples.append(float(tangent[5]))

            if (
                not np.all(np.isfinite(data.qpos))
                or not np.all(np.isfinite(data.qvel))
                or abs(float(state.pitch)) > 1.5
                or float(state.base_position[2]) < 0.02
            ):
                failed = True
                break

        tail_wheel_vel = float(abs(np.mean(wheel_vel_samples))) if wheel_vel_samples else float("nan")
        tail_pos_drift = (
            float(abs(wheel_pos_samples[-1] - wheel_pos_samples[0]))
            if len(wheel_pos_samples) >= 2
            else float("nan")
        )

        print("startup drift")
        print(f"  duration={DURATION_SECONDS:.1f}s tail={TAIL_SECONDS:.1f}s")
        print(f"  tail_abs_mean_wheel_vel={tail_wheel_vel:.4f} m/s")
        print(f"  tail_abs_wheel_pos_drift={tail_pos_drift:.4f} m")
        print(f"  fell={failed}")

        passed = (
            not failed
            and tail_wheel_vel < MAX_TAIL_WHEEL_VEL
            and tail_pos_drift < MAX_TAIL_WHEEL_POS_DRIFT
        )
        print(f"  result={'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
