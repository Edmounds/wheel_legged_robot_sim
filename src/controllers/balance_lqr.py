from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from src.controllers.lqr import solve_discrete_lqr
from src.model_semantics import MODEL_SEMANTICS, WHEEL_FORWARD_SIGNS, WHEEL_RADIUS
from src.state import model_addresses


LEG_ROLL_DIFF_SIGNS: dict[str, float] = {
    MODEL_SEMANTICS.leg_motor_joints[0]: -1.0,
    MODEL_SEMANTICS.leg_motor_joints[1]: 1.0,
}


def _leg_common_to_pitch_coupling(model: Any, data: Any) -> float:
    """实测 common-mode leg ctrl 对 base pitch_rate 的耦合系数 (rad/s² per N·m·leg)。

    在当前 (qpos, qvel) snapshot 下 mj_forward 两次: ctrl=0 baseline 与 ctrl=(L:+1, R:+1)。
    取 base pitch dof (root_qvel + 3) 的 qacc 差, 转换为 pitch_rate (= -wx) 加速度。

    返回值约定: τ_L = τ_R = c 时, base pitch_rate_acc ≈ coupling * c。
    实测在 standing pose ≈ -0.15 rad/s² per N·m·leg (两腿同号 +1 → body 向后仰)。

    不修改 data 状态 (snapshot/restore)。
    """
    addresses = model_addresses(model)
    qpos_save = np.array(data.qpos, copy=True)
    qvel_save = np.array(data.qvel, copy=True)
    ctrl_save = np.array(data.ctrl, copy=True)

    try:
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        pitch_dof = addresses.root_qvel + 3
        qacc_pitch_base = float(data.qacc[pitch_dof])

        for joint_name in MODEL_SEMANTICS.leg_motor_joints:
            data.ctrl[addresses.actuators[joint_name]] = 1.0
        mujoco.mj_forward(model, data)
        qacc_pitch_pulse = float(data.qacc[pitch_dof])
    finally:
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        data.ctrl[:] = ctrl_save
        mujoco.mj_forward(model, data)

    # state.pitch_rate 符号约定: pitch_rate = -wx (见 sim/state.py).
    pitch_rate_acc = -(qacc_pitch_pulse - qacc_pitch_base)
    if not np.isfinite(pitch_rate_acc):
        raise ValueError("leg-common to pitch coupling probe produced non-finite result")
    return pitch_rate_acc


def wheel_ff_gain_for_leg_common(model: Any, data: Any) -> float:
    """计算 wheel forward virtual torque 的 feedforward 增益, 用于抵消 VMC 共模 leg
    torque 引起的 pitch 扰动。

    原理:
      continuous pitch_rate_acc 来自 leg common: coupling * τ_leg_common
      continuous pitch_rate_acc 来自 wheel:      -1/(R*M*L) * τ_wheel_forward
      抵消方程: τ_wheel_forward = coupling * τ_leg_common * R * M * L

    返回 gain 使得 τ_wheel_ff = gain * τ_leg_common_average。
    """
    coupling = _leg_common_to_pitch_coupling(model, data)
    pendulum_length = _com_height_above_wheels(model, data)
    effective_mass = max(float(np.sum(model.body_mass)), 1e-6)
    gain = coupling * WHEEL_RADIUS * effective_mass * pendulum_length
    if not np.isfinite(gain):
        raise ValueError("wheel feedforward gain non-finite")
    return gain


def compute_balance_lqr_gain(
    model: Any,
    data: Any,
    q_diag: np.ndarray,
    r_diag: np.ndarray,
) -> np.ndarray:
    q_diag = _validate_positive_diag(q_diag, (6,), "q_diag")
    r_diag = _validate_positive_diag(r_diag, (2,), "r_diag")

    a, b = _reduced_balance_system(model, data)
    q = np.diag(q_diag)
    r = np.diag(r_diag)
    try:
        gain = solve_discrete_lqr(a, b, q, r)
    except np.linalg.LinAlgError as exc:
        raise ValueError("failed to solve balance LQR") from exc
    if gain.shape != (2, 6) or not np.all(np.isfinite(gain)):
        raise ValueError("balance LQR gain must be finite shape (2, 6)")
    return gain


def compute_balance_lqr_gain_5d(
    model: Any,
    data: Any,
    q_diag: np.ndarray,
    r_diag: np.ndarray,
) -> np.ndarray:
    """5 维平衡 LQR 增益。State: [pitch, pitch_rate, roll, roll_rate, wheel_vel]。

    位置 wheel_pos 不在 LQR state 中——避免位置反馈在 stand 模式下与平衡
    所需的轮子自由运动产生正反馈。位置漂移由外环处理（target_velocity
    或可选的慢速 position anchor）。
    """
    q_diag = _validate_positive_diag(q_diag, (5,), "q_diag")
    r_diag = _validate_positive_diag(r_diag, (2,), "r_diag")

    a, b = _reduced_balance_system_5d(model, data)
    q = np.diag(q_diag)
    r = np.diag(r_diag)
    try:
        gain = solve_discrete_lqr(a, b, q, r)
    except np.linalg.LinAlgError as exc:
        raise ValueError("failed to solve balance LQR") from exc
    if gain.shape != (2, 5) or not np.all(np.isfinite(gain)):
        raise ValueError("5D balance LQR gain must be finite shape (2, 5)")
    return gain


def _reduced_balance_system(model: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_forward(model, data)
    pendulum_length = _com_height_above_wheels(model, data)
    effective_mass = max(float(np.sum(model.body_mass)), 1e-6)
    track_width = _track_width(model, data)
    roll_inertia = _base_roll_inertia(model, data)
    g = 9.81

    # The control loop runs at 500Hz (dt=0.002), while MuJoCo steps at 2000Hz.
    control_dt = 0.002
    dt = control_dt

    b_vel = (1.0 / WHEEL_RADIUS) / effective_mass * dt

    a = np.zeros((6, 6))
    a[0, 0] = 1.0
    a[0, 1] = dt
    a[1, 0] = (g / pendulum_length) * dt
    a[1, 1] = 1.0
    a[2, 2] = 1.0
    a[2, 3] = dt
    a[3, 2] = (g / pendulum_length) * dt
    a[3, 3] = 1.0
    a[4, 4] = 1.0
    a[4, 5] = dt
    a[5, 5] = 1.0

    b = np.zeros((6, 2))
    b[1, 0] = -(b_vel / pendulum_length)
    b[5, 0] = b_vel
    b[3, 1] = (track_width / 2.0) / (roll_inertia * pendulum_length) * dt

    if a.shape != (6, 6) or b.shape != (6, 2) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("reduced balance system must be finite with shapes (6, 6) and (6, 2)")
    return a, b


def _reduced_balance_system_5d(model: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    """5D 状态线性化系统。State: [pitch, pitch_rate, roll, roll_rate, wheel_vel]。

    与 6D 的区别: 删掉 wheel_pos 行/列。wheel_pos 是 wheel_vel 的积分，
    不在 LQR 反馈中。
    """
    mujoco.mj_forward(model, data)
    pendulum_length = _com_height_above_wheels(model, data)
    effective_mass = max(float(np.sum(model.body_mass)), 1e-6)
    track_width = _track_width(model, data)
    roll_inertia = _base_roll_inertia(model, data)
    g = 9.81

    control_dt = 0.002
    dt = control_dt
    b_vel = (1.0 / WHEEL_RADIUS) / effective_mass * dt

    a = np.zeros((5, 5))
    a[0, 0] = 1.0
    a[0, 1] = dt
    a[1, 0] = (g / pendulum_length) * dt
    a[1, 1] = 1.0
    a[2, 2] = 1.0
    a[2, 3] = dt
    a[3, 2] = (g / pendulum_length) * dt
    a[3, 3] = 1.0
    a[4, 4] = 1.0  # wheel_vel identity (was a[5,5] in 6D)

    b = np.zeros((5, 2))
    b[1, 0] = -(b_vel / pendulum_length)
    b[4, 0] = b_vel  # was b[5, 0] in 6D
    b[3, 1] = (track_width / 2.0) / (roll_inertia * pendulum_length) * dt

    if a.shape != (5, 5) or b.shape != (5, 2) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("5D reduced balance system must be finite with shapes (5, 5) and (5, 2)")
    return a, b


def _com_height_above_wheels(model: Any, data: Any) -> float:
    mujoco.mj_forward(model, data)
    body_masses = np.asarray(model.body_mass, dtype=float)
    total_mass = float(np.sum(body_masses))
    if total_mass <= 0.0:
        raise ValueError("model mass must be positive")
    com_z = float(np.sum(body_masses * data.xipos[:, 2]) / total_mass)
    wheel_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for body_name in ("wheel_left", "wheel_right")
    ]
    if any(body_id == -1 for body_id in wheel_ids):
        raise ValueError("missing wheel bodies for LQR pendulum geometry")
    wheel_z = float(np.mean([data.xipos[body_id, 2] for body_id in wheel_ids]))
    height = com_z - wheel_z
    if not np.isfinite(height) or abs(height) < 1e-6:
        raise ValueError("invalid CoM height above wheels for LQR")
    return abs(float(height))


def equilibrium_pitch_from_geometry(model: Any, data: Any) -> float:
    """当前关节构型下,CoM 落到 wheel_mid 正上方所需的 pitch 偏置(项目约定)。

    几何推导: 设 CoM 相对 wheel_mid 在本体系下偏移 [_, dy, dz]。绕本体 X 轴
    旋转 θ 后,世界系 Y 分量 = cos(θ)*dy - sin(θ)*dz。稳态 → tan(θ) = dy/dz。
    项目 pitch 约定 (pitch>0=前倾) 与数学旋转角差一个负号,故返回 -atan2(dy, dz)。

    每次调用都重算,跟随腿高度变化。
    """
    mujoco.mj_forward(model, data)
    masses = np.asarray(model.body_mass, dtype=float)
    total_mass = float(masses.sum())
    if total_mass <= 0.0:
        raise ValueError("model mass must be positive")
    com_world = (masses[:, None] * np.asarray(data.xipos)).sum(axis=0) / total_mass
    wheel_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for body_name in ("wheel_left", "wheel_right")
    ]
    if any(b == -1 for b in wheel_ids):
        raise ValueError("missing wheel bodies for equilibrium pitch")
    wheel_mid = np.mean([data.xipos[b] for b in wheel_ids], axis=0)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id == -1:
        raise ValueError("missing base_link body")
    rotation = np.asarray(data.xmat[base_id]).reshape(3, 3)
    offset_body = rotation.T @ (com_world - wheel_mid)
    dy = float(offset_body[1])
    dz = float(offset_body[2])
    if not np.isfinite(dy) or not np.isfinite(dz) or abs(dz) < 1e-6:
        raise ValueError("invalid CoM/wheel geometry for equilibrium pitch")
    return float(-np.arctan2(dy, dz))


def _track_width(model: Any, data: Any) -> float:
    mujoco.mj_forward(model, data)
    wheel_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for body_name in ("wheel_left", "wheel_right")
    ]
    if any(body_id == -1 for body_id in wheel_ids):
        raise ValueError("missing wheel bodies for LQR track width")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id == -1:
        raise ValueError("missing base_link body for LQR track width")
    base_x_axis = data.xmat[base_id].reshape(3, 3)[:, 0]
    wheel_delta = data.xipos[wheel_ids[0]] - data.xipos[wheel_ids[1]]
    width = abs(float(np.dot(wheel_delta, base_x_axis)))
    if not np.isfinite(width) or width < 1e-6:
        raise ValueError("invalid wheel track width for LQR")
    return width


def _base_roll_inertia(model: Any, data: Any) -> float:
    mujoco.mj_forward(model, data)
    wheel_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for body_name in ("wheel_left", "wheel_right")
    ]
    if any(body_id == -1 for body_id in wheel_ids):
        raise ValueError("missing wheel bodies for LQR roll inertia")
    roll_axis = np.array([0.0, 1.0, 0.0])
    pivot = np.mean([data.xipos[body_id] for body_id in wheel_ids], axis=0)
    inertia = 0.0
    for body_id, mass in enumerate(np.asarray(model.body_mass, dtype=float)):
        if mass <= 0.0:
            continue
        body_inertia = np.asarray(model.body_inertia[body_id], dtype=float)
        inertia += float(np.dot(body_inertia, roll_axis * roll_axis))
        offset = data.xipos[body_id] - pivot
        perpendicular_sq = float(np.dot(offset, offset) - np.dot(offset, roll_axis) ** 2)
        inertia += float(mass) * max(perpendicular_sq, 0.0)
    if not np.isfinite(inertia) or inertia < 1e-9:
        raise ValueError("invalid roll inertia for LQR")
    return inertia


def _balance_state_selection(model: Any) -> np.ndarray:
    addresses = model_addresses(model)
    p = np.zeros((6, 2 * model.nv))
    # 与 extract_sim_state()/balance_tangent_state() 保持一致: pitch 是本体 X 轴倾角（与轮轴平行）。
    pitch_index = addresses.root_qvel + 3
    pitch_rate_index = model.nv + addresses.root_qvel + 3
    roll_index = addresses.root_qvel + 4
    roll_rate_index = model.nv + addresses.root_qvel + 4
    wheel_dof_indices = [addresses.joint_qvel[name] for name in MODEL_SEMANTICS.wheel_joints]

    p[0, pitch_index] = -1.0
    p[1, pitch_rate_index] = -1.0
    p[2, roll_index] = 1.0
    p[3, roll_rate_index] = 1.0
    for joint_name, dof_index in zip(MODEL_SEMANTICS.wheel_joints, wheel_dof_indices):
        sign = WHEEL_FORWARD_SIGNS[joint_name]
        p[4, dof_index] = (sign / len(wheel_dof_indices)) * WHEEL_RADIUS
        p[5, model.nv + dof_index] = (sign / len(wheel_dof_indices)) * WHEEL_RADIUS
    return p


def _lqr_control_selection(model: Any) -> np.ndarray:
    """构建控制选择矩阵 S, 将 2 维虚拟控制映射到执行器空间。

    列 0 是 forward_wheel，列 1 是实测正 roll 加速度方向的 leg diff torque。
    """
    addresses = model_addresses(model)
    s = np.zeros((model.nu, 2))
    for joint_name in MODEL_SEMANTICS.wheel_joints:
        s[addresses.actuators[joint_name], 0] = WHEEL_FORWARD_SIGNS[joint_name]
    for joint_name, sign in LEG_ROLL_DIFF_SIGNS.items():
        s[addresses.actuators[joint_name], 1] = sign
    return s


def _wheel_control_selection(model: Any) -> np.ndarray:
    return _lqr_control_selection(model)


def _validate_positive_diag(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)) or not np.all(array >= 0.0):
        raise ValueError(f"{name} must be finite non-negative shape {shape}")
    return array
