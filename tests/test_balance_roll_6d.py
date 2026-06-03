from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from sim.controllers.balance_lqr import (
    LEG_ROLL_DIFF_SIGNS,
    _balance_state_selection,
    _base_roll_inertia,
    _com_height_above_wheels,
    _lqr_control_selection,
    _reduced_balance_system,
    _track_width,
    compute_balance_lqr_gain,
)
from sim.controllers.balance_state import balance_tangent_state_6d
from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
from sim.controllers.phase import JumpPhase, JumpPhaseMachine
from sim.controllers.jump_trajectory import JumpTrajectory, JumpTrajectoryParams
from sim.model_semantics import MODEL_SEMANTICS, WHEEL_FORWARD_SIGNS, WHEEL_RADIUS
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import extract_sim_state, model_addresses


def _stand_model(tmp_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model_path = prepare_controlled_mujoco_xml(Path("sim/robot/robot.urdf"), cache_dir=tmp_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert stand_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)
    return model, data


def test_balance_tangent_state_6d_matches_sim_state_units(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)
    state = extract_sim_state(model, data)

    tangent = balance_tangent_state_6d(model, data, state)

    left_wheel, right_wheel = MODEL_SEMANTICS.wheel_joints
    expected_wheel_pos = 0.5 * (
        WHEEL_FORWARD_SIGNS[left_wheel] * state.wheel_positions["left"]
        + WHEEL_FORWARD_SIGNS[right_wheel] * state.wheel_positions["right"]
    ) * WHEEL_RADIUS
    expected_wheel_vel = 0.5 * (
        WHEEL_FORWARD_SIGNS[left_wheel] * state.wheel_velocities["left"]
        + WHEEL_FORWARD_SIGNS[right_wheel] * state.wheel_velocities["right"]
    ) * WHEEL_RADIUS

    assert tangent.shape == (6,)
    assert np.allclose(
        tangent,
        np.array([state.pitch, state.pitch_rate, state.roll, state.roll_rate, expected_wheel_pos, expected_wheel_vel]),
    )


def test_balance_state_selection_is_6d_and_matches_small_angle_rows(tmp_path: Path) -> None:
    model, _data = _stand_model(tmp_path)
    addresses = model_addresses(model)

    p = _balance_state_selection(model)

    assert p.shape == (6, 2 * model.nv)
    assert p[0, addresses.root_qvel + 3] == -1.0
    assert p[1, model.nv + addresses.root_qvel + 3] == -1.0
    assert p[2, addresses.root_qvel + 4] == 1.0
    assert p[3, model.nv + addresses.root_qvel + 4] == 1.0
    for joint_name in MODEL_SEMANTICS.wheel_joints:
        dof = addresses.joint_qvel[joint_name]
        assert p[4, dof] == WHEEL_FORWARD_SIGNS[joint_name] * WHEEL_RADIUS / 2.0
        assert p[5, model.nv + dof] == WHEEL_FORWARD_SIGNS[joint_name] * WHEEL_RADIUS / 2.0


def test_lqr_control_selection_maps_forward_and_roll_diff(tmp_path: Path) -> None:
    model, _data = _stand_model(tmp_path)
    addresses = model_addresses(model)

    s = _lqr_control_selection(model)

    assert s.shape == (model.nu, 2)
    for joint_name in MODEL_SEMANTICS.wheel_joints:
        assert s[addresses.actuators[joint_name], 0] == WHEEL_FORWARD_SIGNS[joint_name]
        assert s[addresses.actuators[joint_name], 1] == 0.0
    for joint_name, sign in LEG_ROLL_DIFF_SIGNS.items():
        assert s[addresses.actuators[joint_name], 0] == 0.0
        assert s[addresses.actuators[joint_name], 1] == sign


def test_reduced_balance_system_is_6d_2input_and_stabilizable(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)

    a, b = _reduced_balance_system(model, data)
    # 5D 重构后 STAND_PARAMS.q_diag 是 5 维，这里手动给 6D 测原有 6D 路径
    q_diag_6d = np.array([1000.0, 200.0, 1000.0, 200.0, 50.0, 500.0])
    r_diag_6d = np.array([200.0, 400.0])
    gain = compute_balance_lqr_gain(model, data, q_diag_6d, r_diag_6d)
    eigenvalues = np.linalg.eigvals(a - b @ gain)

    assert a.shape == (6, 6)
    assert b.shape == (6, 2)
    assert gain.shape == (2, 6)
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(b))
    assert np.max(np.abs(eigenvalues)) < 1.0
    assert np.allclose(a[0, 1], 0.002)
    assert np.allclose(a[2, 3], 0.002)
    assert np.allclose(a[4, 5], 0.002)
    assert b[1, 0] < 0.0
    assert b[5, 0] > 0.0
    assert b[3, 1] != 0.0
    assert np.allclose(b[[0, 2, 4], :], 0.0)


def test_reduced_balance_system_matches_roll_formula(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)
    a, b = _reduced_balance_system(model, data)

    dt = 0.002
    l_roll = _com_height_above_wheels(model, data)
    i_roll = _base_roll_inertia(model, data)
    track_width = _track_width(model, data)
    expected_b_roll = (track_width / 2.0) / (i_roll * l_roll) * dt

    assert np.isclose(a[3, 2], (9.81 / l_roll) * dt)
    assert np.isclose(b[3, 1], expected_b_roll)


def test_vmc_mean_and_lqr_roll_diff_are_added_orthogonally(tmp_path: Path) -> None:
    """STAND 模式下 VMC 共模腿力矩与 LQR roll diff 在 leg motor 上线性相加。"""
    model, _data = _stand_model(tmp_path)
    addresses = model_addresses(model)
    controller = CombinedController(STAND_PARAMS)
    vmc_control = np.zeros(model.nu)
    left_leg, right_leg = MODEL_SEMANTICS.leg_motor_joints
    vmc_control[addresses.actuators[left_leg]] = 3.0
    vmc_control[addresses.actuators[right_leg]] = 3.0

    # _allocate_balance_control 写 LQR 部分 (forward + roll + yaw),_merge_vmc_and_clip
    # 把 VMC 共模力矩加到 leg motor 上并按相位 clip。STAND clip = ±3.5。
    lqr_part = controller._allocate_balance_control(model, 0.0, 0.5, 0.0, addresses)
    control = controller._merge_vmc_and_clip(model, lqr_part, vmc_control, addresses, JumpPhase.STAND)

    assert np.isclose(control[addresses.actuators[left_leg]], 2.5)
    assert np.isclose(control[addresses.actuators[right_leg]], 3.5)


def test_positive_roll_virtual_torque_matches_measured_short_step_direction(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)
    addresses = model_addresses(model)
    controller = CombinedController(STAND_PARAMS)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    before = extract_sim_state(model, data)

    control = controller._allocate_balance_control(model, 0.0, 1.0, 0.0, addresses)
    for _ in range(5):
        data.ctrl[:] = 0.0
        for joint_name in MODEL_SEMANTICS.leg_motor_joints:
            act_idx = addresses.actuators[joint_name]
            data.ctrl[act_idx] = control[act_idx]
        mujoco.mj_step(model, data)

    after = extract_sim_state(model, data)
    assert stand_id >= 0
    assert after.roll - before.roll > 0.0


def test_stand_lqr_roll_diff_does_not_write_leg_actuators(tmp_path: Path) -> None:
    from dataclasses import replace

    model, data = _stand_model(tmp_path)
    addresses = model_addresses(model)
    controller = CombinedController(STAND_PARAMS)
    state = extract_sim_state(model, data)
    left_leg, right_leg = MODEL_SEMANTICS.leg_motor_joints

    roll_error = replace(
        state,
        roll=0.1,
        roll_rate=0.0,
        contact_count=2,
    )
    vmc_control = np.zeros(model.nu)
    control = controller._stand_control(
        model,
        data,
        roll_error,
        float(model.opt.timestep),
        vmc_control,
        addresses,
    )

    assert control[addresses.actuators[left_leg]] == 0.0
    assert control[addresses.actuators[right_leg]] == 0.0


def test_airborne_zero_wheel_forward_torque_regardless_of_phase(tmp_path: Path) -> None:
    """contact_count==0 时,LQR 不应在两轮上产生同向 forward 力矩。

    背景: 空中无地面阻尼,持续轮力矩会把轮子加速到 50+ rad/s。重新接地瞬间
    这股切向速度通过滑动摩擦在 CoM 之下产生反向 pitch 反扭,把车架掀翻
    (2026-05-20 run 失败原因)。失败场景不仅是 FLIGHT 相位本身,还包括 LAND
    期间弹跳短暂离地 → 故按"是否接地"判定。

    本测试钉死:
        contact_count=0 时 LQR forward torque = 0 (即使 phase=LAND 或 STAND);
        contact_count>0 时 LQR forward torque 应非零 (基线 sanity).
    """
    from dataclasses import replace

    model, data = _stand_model(tmp_path)
    addresses = model_addresses(model)
    phase_machine = JumpPhaseMachine()
    controller = CombinedController(STAND_PARAMS, phase_machine=phase_machine)

    base_state = extract_sim_state(model, data)
    left_wheel, right_wheel = MODEL_SEMANTICS.wheel_joints

    def forward_mean(control: np.ndarray) -> float:
        left = control[addresses.actuators[left_wheel]] * WHEEL_FORWARD_SIGNS[left_wheel]
        right = control[addresses.actuators[right_wheel]] * WHEEL_FORWARD_SIGNS[right_wheel]
        return 0.5 * (left + right)

    # 1) 接地基线: pitch 误差应该驱动非零 forward torque
    grounded = replace(base_state, pitch=0.1, pitch_rate=0.0, contact_count=2)
    phase_machine.phase = JumpPhase.STAND
    grounded_forward = forward_mean(controller(model, data, grounded))
    assert abs(grounded_forward) > 0.05, (
        f"baseline sanity: grounded with pitch error should drive forward torque, got {grounded_forward}"
    )

    # 2) STAND 下接触不足: 同样的 pitch 误差也应被钳到 0。上坡接触变轻时
    # contact_count 会短暂变成 0/1，继续打轮力矩会让轮子空转，重新接触后
    # 通过切向摩擦冲击 pitch。
    stand_airborne = replace(
        base_state,
        pitch=0.1,
        pitch_rate=0.0,
        contact_count=0,
    )
    phase_machine.phase = JumpPhase.STAND
    stand_airborne_forward = forward_mean(controller(model, data, stand_airborne))
    assert stand_airborne_forward == 0.0, (
        f"STAND (contact=0) must zero forward wheel torque, got {stand_airborne_forward}"
    )

    stand_single_contact = replace(
        base_state,
        pitch=0.1,
        pitch_rate=0.0,
        contact_count=1,
    )
    phase_machine.phase = JumpPhase.STAND
    stand_single_contact_forward = forward_mean(controller(model, data, stand_single_contact))
    assert stand_single_contact_forward == 0.0, (
        f"STAND (contact=1) must zero forward wheel torque, got {stand_single_contact_forward}"
    )

    # 3) 空中 (FLIGHT): 同样的 pitch 误差应被钳到 0
    in_air_velocity = base_state.base_linear_velocity.copy()
    in_air_velocity[2] = 1.0  # 让 phase_machine 不会立刻从 FLIGHT 跳成 LAND
    airborne = replace(
        base_state,
        pitch=0.1,
        pitch_rate=0.0,
        base_linear_velocity=in_air_velocity,
        contact_count=0,
    )
    traj = JumpTrajectory(JumpTrajectoryParams(), h_start=0.142, cmd_jump_amplitude=1.0)
    phase_machine.start_jump(traj)
    phase_machine.phase = JumpPhase.FLIGHT
    flight_forward = forward_mean(controller(model, data, airborne))
    assert flight_forward == 0.0, (
        f"FLIGHT (contact=0) must zero forward wheel torque, got {flight_forward}"
    )

    # 4) LAND 期间弹跳离地 (contact=0 但 phase=LAND): 同样应钳到 0
    # 这一案例覆盖 2026-05-20 用户 run 的真实失败模式: 第二次跳的 LAND 阶段
    # 多次 contact=0/2 振荡,期间 LQR 把轮速从 +37 推到 -433 rad/s 直接掀翻。
    traj.setup_land(h_contact=0.10, v_contact=-0.5, h_target=0.142)
    phase_machine.phase = JumpPhase.LAND
    land_airborne_forward = forward_mean(controller(model, data, airborne))
    assert land_airborne_forward == 0.0, (
        f"LAND bounce (contact=0) must zero forward wheel torque, got {land_airborne_forward}"
    )
