from __future__ import annotations

import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.model_xml import _configure_geom_collisions, _ensure_equality_constraints, _urdf_to_mjcf
from sim.mujoco_mesh_preprocess import prepare_mujoco_xml


LEG_MOTOR_JOINTS = ("base_link_旋转-2", "base_link_旋转-1")
LEG_JOINTS_LEFT = ("base_link_旋转-2", "base_link_旋转-4", "link1_left_旋转-6")
LEG_JOINTS_RIGHT = ("base_link_旋转-1", "base_link_旋转-3", "link1_right_旋转-5")
JOINT_PAIRS = (
    ("base_link_旋转-2", "base_link_旋转-1"),
    ("base_link_旋转-4", "base_link_旋转-3"),
    ("link1_left_旋转-6", "link1_right_旋转-5"),
    ("link2_left_旋转-13", "link2_right_旋转-12"),
)


def _build_position_model(tmp_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    converted_xml = _urdf_to_mjcf(Path("sim/robot/robot.urdf").resolve(), tmp_path)
    prepared_xml = prepare_mujoco_xml(converted_xml)
    tree = ET.parse(prepared_xml)
    root = tree.getroot()

    actuator = ET.SubElement(root, "actuator")
    for joint_name in LEG_MOTOR_JOINTS:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"act_{joint_name}",
                "joint": joint_name,
                "kp": "5",
                "ctrllimited": "true",
                "ctrlrange": "-1.5 1.5",
                "gear": "1.0",
            },
        )

    option = root.find("option")
    if option is None:
        option = ET.Element("option", {"gravity": "0 0 0"})
        root.insert(0, option)
    else:
        option.set("gravity", "0 0 0")

    _configure_geom_collisions(root)
    _ensure_equality_constraints(root)

    out_xml = tmp_path / "check_model.xml"
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(out_xml))
    data = mujoco.MjData(model)
    for joint_name in LEG_MOTOR_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = 0.5
    mujoco.mj_forward(model, data)
    return model, data


def _joint_qpos_addresses(model: mujoco.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in joint_names]
    return [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]


def _run_synchronized_position_rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl_value: float = 1.0,
    duration: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    left_qpos = _joint_qpos_addresses(model, LEG_JOINTS_LEFT)
    right_qpos = _joint_qpos_addresses(model, LEG_JOINTS_RIGHT)
    left_history = []
    right_history = []

    for _step in range(int(duration / model.opt.timestep)):
        data.ctrl[:] = ctrl_value
        mujoco.mj_step(model, data)
        left_history.append([data.qpos[address] for address in left_qpos])
        right_history.append([data.qpos[address] for address in right_qpos])

    return np.array(left_history), np.array(right_history)


def test_synchronized_leg_position_actuators_do_not_need_right_gear_compensation(tmp_path: Path) -> None:
    model, data = _build_position_model(tmp_path)
    # ctrl=0.3 远在 motor joint upper=+0.60 之内,避开限位反弹引入的瞬态不对称。
    # 较小目标值也避免无阻尼 position 控制器 overshoot 时撞到限位。
    left_history, right_history = _run_synchronized_position_rollout(model, data, ctrl_value=0.3)

    assert np.all(data.ctrl == 0.3)
    assert np.max(np.abs(left_history[:, 0] - right_history[:, 0])) < 0.02


def test_leg_joint_world_axes_match_common_physical_semantics(tmp_path: Path) -> None:
    model, data = _build_position_model(tmp_path)

    for left_joint, right_joint in JOINT_PAIRS:
        left_axis = _joint_world_axis(model, data, left_joint)
        right_axis = _joint_world_axis(model, data, right_joint)
        assert np.allclose(left_axis, right_axis, atol=1e-6)


def test_mirrored_leg_body_frames_remain_right_handed(tmp_path: Path) -> None:
    model, data = _build_position_model(tmp_path)

    for body_name in ("link1_right", "link2_right", "link3_right", "wheel_right"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_rot = data.xmat[body_id].reshape(3, 3)
        assert np.isclose(np.linalg.det(body_rot), 1.0, atol=1e-9)


def _joint_world_axis(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> np.ndarray:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    body_id = model.jnt_bodyid[joint_id]
    body_rot = data.xmat[body_id].reshape(3, 3)
    return body_rot @ model.jnt_axis[joint_id]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_, data_ = _build_position_model(Path(tmp_dir))
        left, right = _run_synchronized_position_rollout(model_, data_)
        print("Max abs trajectory error:", np.max(np.abs(left - right), axis=0))
