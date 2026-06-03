from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from src.model_semantics import MODEL_SEMANTICS


@dataclass(frozen=True)
class ModelAddresses:
    root_qpos: int
    root_qvel: int
    joint_qpos: dict[str, int]
    joint_qvel: dict[str, int]
    actuators: dict[str, int]


@dataclass(frozen=True)
class SimState:
    base_position: np.ndarray
    base_quaternion: np.ndarray
    base_linear_velocity: np.ndarray
    base_angular_velocity: np.ndarray
    pitch: float
    pitch_rate: float
    roll: float
    roll_rate: float
    wheel_positions: dict[str, float]
    wheel_velocities: dict[str, float]
    leg_joint_positions: dict[str, float]
    leg_joint_velocities: dict[str, float]
    contact_count: int


def model_addresses(model: mujoco.MjModel) -> ModelAddresses:
    cached = _MODEL_ADDRESSES_CACHE.get(id(model))
    if cached is not None:
        return cached

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root_id == -1:
        raise ValueError("missing root freejoint")

    joint_qpos: dict[str, int] = {}
    joint_qvel: dict[str, int] = {}
    for joint_name in MODEL_SEMANTICS.joint_roles:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"missing joint: {joint_name}")
        joint_qpos[joint_name] = int(model.jnt_qposadr[joint_id])
        joint_qvel[joint_name] = int(model.jnt_dofadr[joint_id])

    actuators: dict[str, int] = {}
    for joint_name in MODEL_SEMANTICS.wheel_joints + MODEL_SEMANTICS.leg_motor_joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _actuator_for_joint(model, joint_id)
        if actuator_id == -1:
            raise ValueError(f"missing actuator for joint: {joint_name}")
        actuators[joint_name] = actuator_id

    result = ModelAddresses(
        root_qpos=int(model.jnt_qposadr[root_id]),
        root_qvel=int(model.jnt_dofadr[root_id]),
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        actuators=actuators,
    )
    _MODEL_ADDRESSES_CACHE[id(model)] = result
    return result


_MODEL_ADDRESSES_CACHE: dict[int, ModelAddresses] = {}
_BODY_ID_CACHE: dict[tuple[int, str], int] = {}
_ACTUATOR_ID_CACHE: dict[tuple[int, str], int] = {}
_EQUALITY_ID_CACHE: dict[tuple[int, str], int] = {}


def body_id(model: mujoco.MjModel, name: str) -> int:
    key = (id(model), name)
    cached = _BODY_ID_CACHE.get(key)
    if cached is not None:
        return cached
    resolved = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if resolved == -1:
        raise ValueError(f"missing body: {name}")
    _BODY_ID_CACHE[key] = resolved
    return resolved


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    """Cached actuator-id lookup. Returns -1 if not found (caller decides)."""
    key = (id(model), name)
    cached = _ACTUATOR_ID_CACHE.get(key)
    if cached is not None:
        return cached
    resolved = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    _ACTUATOR_ID_CACHE[key] = resolved
    return resolved


def equality_id(model: mujoco.MjModel, name: str) -> int:
    key = (id(model), name)
    cached = _EQUALITY_ID_CACHE.get(key)
    if cached is not None:
        return cached
    resolved = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    if resolved == -1:
        raise ValueError(f"missing equality constraint: {name}")
    _EQUALITY_ID_CACHE[key] = resolved
    return resolved


def extract_sim_state(model: mujoco.MjModel, data: mujoco.MjData) -> SimState:
    addresses = model_addresses(model)
    root_qpos = addresses.root_qpos
    root_qvel = addresses.root_qvel

    base_quaternion = np.array(data.qpos[root_qpos + 3 : root_qpos + 7], dtype=float)
    base_angular_velocity = np.array(data.qvel[root_qvel + 3 : root_qvel + 6], dtype=float)
    wheel_names = {"left": MODEL_SEMANTICS.wheel_joints[0], "right": MODEL_SEMANTICS.wheel_joints[1]}

    return SimState(
        base_position=np.array(data.qpos[root_qpos : root_qpos + 3], dtype=float),
        base_quaternion=base_quaternion,
        base_linear_velocity=np.array(data.qvel[root_qvel : root_qvel + 3], dtype=float),
        base_angular_velocity=base_angular_velocity,
        # 绕本体 X 轴的倾倒角 (与轮轴平行)。约定: pitch > 0 = 向前倾 (CoM 向 +Y 方向移)。
        pitch=_pitch_from_quaternion(base_quaternion),
        pitch_rate=-float(base_angular_velocity[0]),
        # 绕本体 Y 轴的左右倾角。约定只要求与 roll_rate 和 LQR 选择矩阵一致。
        roll=_roll_from_quaternion(base_quaternion),
        roll_rate=float(base_angular_velocity[1]),
        wheel_positions={side: float(data.qpos[addresses.joint_qpos[name]]) for side, name in wheel_names.items()},
        wheel_velocities={side: float(data.qvel[addresses.joint_qvel[name]]) for side, name in wheel_names.items()},
        leg_joint_positions={
            name: float(data.qpos[addresses.joint_qpos[name]]) for name in MODEL_SEMANTICS.leg_motor_joints
        },
        leg_joint_velocities={
            name: float(data.qvel[addresses.joint_qvel[name]]) for name in MODEL_SEMANTICS.leg_motor_joints
        },
        contact_count=int(data.ncon),
    )


def _actuator_for_joint(model: mujoco.MjModel, joint_id: int) -> int:
    for actuator_id in range(model.nu):
        if int(model.actuator_trnid[actuator_id, 0]) == joint_id:
            return actuator_id
    return -1


def _pitch_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    # 绕本体 X 轴的旋转 (atan2 形式)。负号使得 pitch > 0 对应"前倾"
    # (CoM 向 +Y 方向倾，与标准倒立摆 LQR 约定一致: 前倾时驱动轮子向前接住)。
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(-np.arctan2(sinr_cosp, cosr_cosp))


def _roll_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    sinp = 2.0 * (w * y - z * x)
    return float(np.arcsin(np.clip(sinp, -1.0, 1.0)))
