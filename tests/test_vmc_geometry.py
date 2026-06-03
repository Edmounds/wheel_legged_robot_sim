from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from sim.controllers.vmc import (
    BASE_BODY_NAME,
    LEG_CLOSED_LOOP,
    VmcController,
    VmcParams,
    _average_leg_height,
    _closed_loop_leg_motor_jacobian,
    _leg_height,
)
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import body_id, model_addresses


def _load_model(tmp_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model_path = prepare_controlled_mujoco_xml(Path("sim/robot/robot.urdf"), cache_dir=tmp_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_leg_height_uses_world_z_distance(tmp_path: Path) -> None:
    model, data = _load_model(tmp_path)
    base = body_id(model, BASE_BODY_NAME)

    heights = {
        side: _leg_height(model, data, geometry.wheel_body)
        for side, geometry in LEG_CLOSED_LOOP.items()
    }

    assert heights["left"] > 0.0
    assert heights["right"] > 0.0
    assert np.isclose(heights["left"], heights["right"], atol=0.01)
    for side, geometry in LEG_CLOSED_LOOP.items():
        wheel = body_id(model, geometry.wheel_body)
        expected = float(data.xipos[base, 2] - data.xipos[wheel, 2])
        assert np.isclose(heights[side], expected, atol=1e-12)


def test_leg_height_is_not_projected_on_base_local_z(tmp_path: Path) -> None:
    model, data = _load_model(tmp_path)
    base = body_id(model, BASE_BODY_NAME)
    original_qpos = data.qpos.copy()

    try:
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 1.0, 0.0]), 0.3)
        data.qpos[3:7] = quat
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        for _side, geometry in LEG_CLOSED_LOOP.items():
            wheel = body_id(model, geometry.wheel_body)
            expected = float(data.xipos[base, 2] - data.xipos[wheel, 2])
            assert np.isclose(_leg_height(model, data, geometry.wheel_body), expected, atol=1e-12)
    finally:
        data.qpos[:] = original_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)


def test_vmc_leg_motor_jacobians_are_symmetric_for_current_model(tmp_path: Path) -> None:
    model, data = _load_model(tmp_path)

    controller = VmcController(VmcParams(nominal_height=0.0, kp_motor=0.0, kd_motor=0.0))
    jacobian = controller.leg_height_jacobian(model, data)

    assert jacobian["left"] > 0.0
    assert jacobian["right"] > 0.0
    assert np.isclose(jacobian["left"], jacobian["right"], rtol=0.0, atol=1e-5)


def test_vmc_slews_nominal_height_and_reports_target_motor_rate(tmp_path: Path) -> None:
    model, data = _load_model(tmp_path)
    controller = VmcController(
        VmcParams(
            nominal_height=0.140,
            kp_motor=0.0,
            kd_motor=0.0,
            max_height_rate=0.01,
        )
    )
    state = _state(model, data)

    controller(model, data, state)
    assert all(rate == 0.0 for rate in controller.last_target_motor_rate.values())

    controller.params.nominal_height = 0.150
    controller(model, data, state)

    assert np.isclose(controller._height_filtered, 0.140 + 0.01 * model.opt.timestep)
    assert controller.last_target_motor_rate["left"] > 0.0
    assert controller.last_target_motor_rate["right"] > 0.0
    assert np.isclose(
        controller.last_target_motor_rate["left"],
        controller.last_target_motor_rate["right"],
        rtol=0.0,
        atol=2e-5,
    )


def test_roll_leveling_uses_stand_roll_rate_without_contact_gate(tmp_path: Path) -> None:
    model, data = _load_model(tmp_path)
    controller = VmcController(
        VmcParams(
            nominal_height=0.142,
            kp_motor=0.0,
            kd_motor=0.0,
            roll_level_kp_height=0.0,
            roll_level_kd_height=0.002,
            roll_level_offset_limit=0.020,
        )
    )
    state = _state(model, data)

    neutral = replace(state, roll=0.0, roll_rate=0.0, contact_count=len(LEG_CLOSED_LOOP))
    controller(model, data, neutral)
    neutral_left = controller.last_target_heights["left"]
    neutral_right = controller.last_target_heights["right"]

    stand_fast_roll = replace(state, roll=0.0, roll_rate=1.0, contact_count=len(LEG_CLOSED_LOOP))
    controller(model, data, stand_fast_roll)
    assert controller.last_target_heights["left"] > neutral_left
    assert controller.last_target_heights["right"] < neutral_right

    single_contact_roll = replace(state, roll=0.0, roll_rate=1.0, contact_count=len(LEG_CLOSED_LOOP) - 1)
    controller(model, data, single_contact_roll)
    assert controller.last_target_heights["left"] > neutral_left
    assert controller.last_target_heights["right"] < neutral_right

    left_wheel = body_id(model, LEG_CLOSED_LOOP["left"].wheel_body)
    data.xipos[left_wheel, 2] += 0.010
    full_contact_roll = replace(state, roll=0.0, roll_rate=0.0, contact_count=len(LEG_CLOSED_LOOP))
    controller(model, data, full_contact_roll)
    assert controller.last_target_heights["left"] < neutral_left
    assert controller.last_target_heights["right"] > neutral_right


def test_closed_loop_leg_motor_jacobian_matches_constrained_finite_difference(tmp_path: Path) -> None:
    model, data = _load_model(tmp_path)

    for side, geometry in LEG_CLOSED_LOOP.items():
        analytic = _closed_loop_leg_motor_jacobian(model, data, geometry)
        numeric = _constrained_leg_height_finite_difference(model, data, side)

        assert np.isclose(analytic, numeric, rtol=0.05, atol=1e-5)


def test_positive_closed_loop_motor_torque_increases_average_leg_height(tmp_path: Path) -> None:
    model, data = _load_model(tmp_path)
    addresses = model_addresses(model)
    model.opt.gravity[:] = 0.0
    mujoco.mj_forward(model, data)

    for side, geometry in LEG_CLOSED_LOOP.items():
        jacobian = _closed_loop_leg_motor_jacobian(model, data, geometry)
        data.ctrl[addresses.actuators[geometry.motor_joint]] = float(np.sign(jacobian))

    initial_height = _average_leg_height(model, data)
    for _ in range(50):
        mujoco.mj_step(model, data)

    assert _average_leg_height(model, data) > initial_height + 0.01


def test_closed_loop_leg_motor_jacobian_rejects_singular_passive_constraint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model, data = _load_model(tmp_path)

    def singular_constraint_jacobian(_model: mujoco.MjModel, _data: mujoco.MjData, _equality_name: str) -> np.ndarray:
        return np.zeros((3, _model.nv))

    monkeypatch.setattr("sim.controllers.vmc._connect_constraint_jacobian", singular_constraint_jacobian)

    with np.testing.assert_raises_regex(ValueError, "closed-loop jacobian singular"):
        _closed_loop_leg_motor_jacobian(model, data, LEG_CLOSED_LOOP["left"])


def test_right_link2_frame_keeps_drive_passive_inertia_coupling_symmetric(tmp_path: Path) -> None:
    model, data = _load_model(tmp_path)

    mass_matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass_matrix, data.qM)

    def coupling(joint_a: str, joint_b: str) -> float:
        id_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_a)
        id_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_b)
        return float(mass_matrix[model.jnt_dofadr[id_a], model.jnt_dofadr[id_b]])

    left = coupling("link1_left_旋转-6", "link2_left_旋转-13")
    right = coupling("link1_right_旋转-5", "link2_right_旋转-12")

    assert left > 0.0
    assert right > 0.0
    assert np.isclose(left, right, rtol=0.01, atol=1e-6)


def test_link23_connect_constraints_start_closed(tmp_path: Path) -> None:
    model, data = _load_model(tmp_path)

    def body_point(body_name: str, local_point: np.ndarray) -> np.ndarray:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local_point

    for side, body1, body2 in (
        ("left", "link2_left", "link3_left"),
        ("right", "link2_right", "link3_right"),
    ):
        equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"link23_{side}_connect")
        point1 = body_point(body1, model.eq_data[equality_id, 0:3])
        point2 = body_point(body2, model.eq_data[equality_id, 3:6])

        assert np.linalg.norm(point1 - point2) < 1e-6


def _constrained_leg_height_finite_difference(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    epsilon: float = 1e-5,
) -> float:
    addresses = model_addresses(model)
    geometry = LEG_CLOSED_LOOP[side]
    active_qpos = addresses.joint_qpos[geometry.motor_joint]
    passive_qpos = [addresses.joint_qpos[name] for name in geometry.passive_joints]
    equality_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, geometry.equality_name)
    if equality_id == -1:
        raise ValueError(f"missing equality constraint: {geometry.equality_name}")

    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    original_ctrl = data.ctrl.copy()

    def height_at(delta: float) -> float:
        def residual(passive_values: np.ndarray) -> np.ndarray:
            data.qpos[:] = original_qpos
            data.qpos[active_qpos] = original_qpos[active_qpos] + delta
            data.qpos[passive_qpos] = passive_values
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            return _connect_residual(model, data, equality_id)

        result = least_squares(
            residual,
            original_qpos[passive_qpos],
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=100,
        )
        data.qpos[:] = original_qpos
        data.qpos[active_qpos] = original_qpos[active_qpos] + delta
        data.qpos[passive_qpos] = result.x
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return _leg_height(model, data, geometry.wheel_body)

    try:
        height_plus = height_at(epsilon)
        height_minus = height_at(-epsilon)
    finally:
        data.qpos[:] = original_qpos
        data.qvel[:] = original_qvel
        data.ctrl[:] = original_ctrl
        mujoco.mj_forward(model, data)

    return float((height_plus - height_minus) / (2.0 * epsilon))


def _connect_residual(model: mujoco.MjModel, data: mujoco.MjData, equality_id: int) -> np.ndarray:
    body_a = int(model.eq_obj1id[equality_id])
    body_b = int(model.eq_obj2id[equality_id])
    point_a = _body_point(data, body_a, model.eq_data[equality_id, 0:3])
    point_b = _body_point(data, body_b, model.eq_data[equality_id, 3:6])
    return point_a - point_b


def _body_point(data: mujoco.MjData, body_id: int, local_point: np.ndarray) -> np.ndarray:
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local_point


def _state(model: mujoco.MjModel, data: mujoco.MjData):
    from sim.state import extract_sim_state

    return extract_sim_state(model, data)
