from __future__ import annotations

import numpy as np

from src.model_semantics import MODEL_SEMANTICS, WHEEL_FORWARD_SIGNS, WHEEL_RADIUS
from src.state import SimState


def balance_tangent_state(model: object, data: object, state: SimState) -> np.ndarray:
    """提取 LQR 平衡控制用的 4 维切线状态。

    状态向量: [pitch(rad), pitch_rate(rad/s), avg_wheel_pos(m), avg_wheel_vel(m/s)]

    pitch 和 pitch_rate 与 sim.state.extract_sim_state 保持一致:
    - pitch 从 MuJoCo 四元数 [w, x, y, z] 提取本体 X 轴的倾倒角 (与轮轴平行)
    - pitch_rate = -base_angular_velocity[0]

    轮子位置/速度使用 WHEEL_FORWARD_SIGNS 统一方向后取平均。
    """
    del model, data

    wheel_names = MODEL_SEMANTICS.wheel_joints
    forward_wheel_positions = [
        WHEEL_FORWARD_SIGNS[name] * state.wheel_positions[_side_key(name)]
        for name in wheel_names
    ]
    forward_wheel_velocities = [
        WHEEL_FORWARD_SIGNS[name] * state.wheel_velocities[_side_key(name)]
        for name in wheel_names
    ]

    tangent = np.array(
        [
            state.pitch,
            state.pitch_rate,
            float(np.mean(forward_wheel_positions)) * WHEEL_RADIUS,
            float(np.mean(forward_wheel_velocities)) * WHEEL_RADIUS,
        ],
        dtype=float,
    )
    if tangent.shape != (4,) or not np.all(np.isfinite(tangent)):
        raise ValueError("balance tangent state must be shape (4,) and finite")
    return tangent


def balance_tangent_state_6d(model: object, data: object, state: SimState) -> np.ndarray:
    """提取 LQR 平衡控制用的 6 维切线状态。

    状态向量: [pitch, pitch_rate, roll, roll_rate, avg_wheel_pos, avg_wheel_vel]
    """
    four_d = balance_tangent_state(model, data, state)
    tangent = np.array(
        [
            four_d[0],
            four_d[1],
            state.roll,
            state.roll_rate,
            four_d[2],
            four_d[3],
        ],
        dtype=float,
    )
    if tangent.shape != (6,) or not np.all(np.isfinite(tangent)):
        raise ValueError("6D balance tangent state must be shape (6,) and finite")
    return tangent


def balance_tangent_state_5d(model: object, data: object, state: SimState) -> np.ndarray:
    """提取 LQR 平衡控制用的 5 维切线状态。

    状态向量: [pitch, pitch_rate, roll, roll_rate, avg_wheel_vel]

    与 6D 的区别: 不含 wheel_pos。LQR 内环只跟踪平衡相关的 5 维；
    位置漂移由外环（target_velocity 或慢速 position anchor）处理，避免
    位置锁在冲击/震荡瞬态与轮子运动形成正反馈。
    """
    four_d = balance_tangent_state(model, data, state)
    tangent = np.array(
        [
            four_d[0],
            four_d[1],
            state.roll,
            state.roll_rate,
            four_d[3],
        ],
        dtype=float,
    )
    if tangent.shape != (5,) or not np.all(np.isfinite(tangent)):
        raise ValueError("5D balance tangent state must be shape (5,) and finite")
    return tangent


def _side_key(joint_name: str) -> str:
    """将轮子关节名映射到 SimState 中的 left/right 键。"""
    if joint_name == MODEL_SEMANTICS.wheel_joints[0]:
        return "left"
    return "right"
