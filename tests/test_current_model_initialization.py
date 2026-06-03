from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from sim.model_xml import prepare_controlled_mujoco_xml


def test_controlled_model_has_stand_keyframe_with_constraint_consistent_pose(tmp_path: Path) -> None:
    model_path = prepare_controlled_mujoco_xml(Path("sim/robot/robot.urdf"), cache_dir=tmp_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert stand_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)

    # Wheels symmetric and on the floor (z=0) within proxy radius
    wl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_left")
    wr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wheel_right")
    assert np.isclose(data.xpos[wl, 2], data.xpos[wr, 2], atol=1e-3)
    # Base above floor
    assert data.qpos[2] > 0.1
    # Identity orientation at keyframe load
    assert np.allclose(data.qpos[3:7], np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
