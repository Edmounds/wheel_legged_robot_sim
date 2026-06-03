from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from src.controllers.leg_height_lut import LegHeightLUT
from src.controllers.phase import JumpPhase, JumpPhaseMachine
from src.state import SimState, body_id, equality_id, model_addresses

BASE_BODY_NAME = "base_link"
LEG_JACOBIAN_SINGULAR_VALUE_EPS = 1e-8


@dataclass(frozen=True)
class LegClosedLoopGeometry:
    side: str
    motor_joint: str
    passive_joints: tuple[str, str]
    equality_name: str
    wheel_body: str


LEG_CLOSED_LOOP: dict[str, LegClosedLoopGeometry] = {
    "left": LegClosedLoopGeometry(
        side="left",
        motor_joint="base_link_旋转-2",
        passive_joints=("base_link_旋转-4", "link1_left_旋转-6"),
        equality_name="link23_left_connect",
        wheel_body="wheel_left",
    ),
    "right": LegClosedLoopGeometry(
        side="right",
        motor_joint="base_link_旋转-1",
        passive_joints=("base_link_旋转-3", "link1_right_旋转-5"),
        equality_name="link23_right_connect",
        wheel_body="wheel_right",
    ),
}


@dataclass
class VmcParams:
    """VMC leg-height + jump-phase controller parameters.

    调参顺序 (实机/headless 都建议从上往下冻结):
      1. nominal_height            — stand 站姿高度 (m), 必须在 LUT 范围内
      2. kp_motor / kd_motor       — joint-space PD on active motor angle
                                     (STAND/CROUCH/EXTEND 通用)
      3. kp_land  / kd_land        — LAND 阶段独立 PD, 比 STAND 软 P 硬 D 吸冲击
      4. flight_pitch_kd           — FLIGHT 期间对称腿 motor 上的 pitch_rate 阻尼
      5. max_height_rate           — nominal_height 步进上限 (m/s)
      6. roll_level_*              — STAND 斜坡找平偏置 (默认 0, 仅 ramp 实验启用)
    """

    # nominal_height is runtime-writable so the cmd_height slider can drive VMC.
    nominal_height: float
    # Joint-space PID gains on the active leg motor angles. Combined with the
    # LUT-driven target motor angle, these replace the old task-space height
    # spring which was unstable near the four-bar singularity.
    kp_motor: float
    kd_motor: float
    # LAND 阶段独立的 PD 增益 (绝对值, 不是 STAND 增益的缩放). 落地需要软 P + 强 D
    # 吸收冲击, 跟 STAND 跟踪目标的需求不同. 用绝对值消除"调 STAND 同时改 LAND"
    # 的隐式耦合.
    kp_land: float = 15.0
    kd_land: float = 3.5
    max_height_rate: float = 0.1
    # FLIGHT 期间对称腿 motor 上的 pitch_rate 阻尼增益 (N·m per rad/s).
    # 见 _control() FLIGHT 分支注释 — 仅 D 项, 加 P 会与 leg-gravity 偏置形成正反馈.
    flight_pitch_kd: float = 1.5
    # STAND 斜坡找平: 测左右轮高差做前馈, 调左右腿高差把 base 调平 (纯前馈, kp 默认 0).
    # roll_level_offset_limit 是左右腿高差的单侧上限 (m); 0 = off.
    roll_level_kp_height: float = 0.0
    roll_level_kd_height: float = 0.0
    roll_level_offset_limit: float = 0.0
    # 上坡降站高: 找平需要大高差时降低共模站高, 给"伸长"那条腿留出 LUT 余量.
    # nominal 142mm 离 h_max 153.7mm 只有 ~12mm, 不降站高就伸不出去调平大高差.
    # 纯前馈, 由当前 roll_level 偏置量驱动. 值 = 伸长腿离 h_max 至少保留的余量 (m); 0 = off.
    slope_squat_margin: float = 0.0
    lut: LegHeightLUT = field(default_factory=LegHeightLUT.from_json)


class VmcController:
    JACOBIAN_REFRESH_PERIOD = 10  # recompute leg motor jacobian every N control steps

    def __init__(self, params: VmcParams, phase_machine: JumpPhaseMachine | None = None) -> None:
        self.params = params
        self.phase_machine = phase_machine
        self._neutral_height_offsets: dict[int, dict[str, float]] = {}
        self._motor_jacobian_cache: dict[str, float] | None = None
        self._height_jacobian_rows_cache: dict[str, np.ndarray] | None = None
        self._motor_jacobian_step_count: int = 0
        self._height_filtered: float | None = None
        self._crouch_start_height: float | None = None
        self._target_motor_angle_prev: dict[str, float] = {}
        self.last_target_motor_rate: dict[str, float] = {side: 0.0 for side in LEG_CLOSED_LOOP}
        self.last_target_heights: dict[str, float] = {side: 0.0 for side in LEG_CLOSED_LOOP}

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, state: SimState) -> np.ndarray:
        return self._control(model, data, state, update_phase=True)

    def preview_control(self, model: mujoco.MjModel, data: mujoco.MjData, state: SimState) -> np.ndarray:
        """Compute VMC output without advancing the jump phase machine."""
        return self._control(model, data, state, update_phase=False)

    def leg_height_jacobian(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        """Return dh/dq for each active leg motor using the current MuJoCo geometry."""
        return {
            side: _closed_loop_leg_motor_jacobian(model, data, geometry)
            for side, geometry in LEG_CLOSED_LOOP.items()
        }

    def _control(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state: SimState,
        *,
        update_phase: bool,
    ) -> np.ndarray:
        params_values = np.array([
            self.params.nominal_height,
            self.params.kp_motor,
            self.params.kd_motor,
            self.params.kp_land,
            self.params.kd_land,
            self.params.max_height_rate,
            self.params.flight_pitch_kd,
            self.params.roll_level_kp_height,
            self.params.roll_level_kd_height,
            self.params.roll_level_offset_limit,
            self.params.slope_squat_margin,
        ])
        if not np.all(np.isfinite(params_values)):
            raise ValueError("VMC params must be finite")
        if not np.all(np.isfinite(state.base_position)) or not np.all(np.isfinite(state.base_linear_velocity)):
            raise ValueError("VMC state must be finite")

        phase = self.phase_machine.phase if self.phase_machine is not None else JumpPhase.STAND
        current_leg_height = _average_leg_height(model, data)
        if self.phase_machine is not None and update_phase:
            self.phase_machine.update(
                dt=float(model.opt.timestep),
                leg_height=current_leg_height,
                vz=float(state.base_linear_velocity[2]),
                contact_count=state.contact_count,
                pitch=float(state.pitch),
            )
            phase = self.phase_machine.phase

        # FLIGHT: 用对称腿电机做 pitch_rate 反作用阻尼。
        # 物理依据: 电机力矩 τ 给腿 → Newton 3rd 给 base 反作用 -τ (绕 pitch 轴)。
        # 左右两个电机同向施加 τ → base 受 -2τ 的 pitch 力矩,可主动减速 pitch_rate。
        # (旧实现返回 0 是"干净"但起跳后 pitch_rate 累积 -4 rad/s,250ms 飞行翻 1 rad)
        #
        # 注意: 必须严格对称 (左右同 τ),否则会产生 roll 力矩。
        # 不对 height/位置做反馈 (legs 在空中无 ground reaction,kp 没意义);
        # 只做 pitch_rate 一维阻尼。
        # FALLEN: 完全 0,不再控制。
        if phase == JumpPhase.FALLEN:
            self.last_target_motor_rate = {side: 0.0 for side in LEG_CLOSED_LOOP}
            return np.zeros(model.nu)
        if phase == JumpPhase.FLIGHT:
            self.last_target_motor_rate = {side: 0.0 for side in LEG_CLOSED_LOOP}
            trajectory = self.phase_machine.trajectory if self.phase_machine is not None else None
            lock_height = float(trajectory.h_high) if trajectory is not None else float(self.params.nominal_height)
            self.last_target_heights = {side: lock_height for side in LEG_CLOSED_LOOP}
            # 空中姿态: 仅 pitch_rate 阻尼 (PD 中不加 K_p 位置项)。
            # τ_motor = -K_d * pitch_rate, 两腿对称同号 → 通过 four-bar 给 base 反向
            # pitch 力矩。flight_pitch_kd 默认 1.5, clip ±3.5 N·m 是实测稳定点。
            #
            # 为什么不加 K_p:
            # FLIGHT 期间 leg gravity 摆动 + EXTEND 末期 leg 角动量回流给 base 一个
            # 恒定 ~+2.3 N·m 的偏置 pitch 力矩, 让 pitch_rate 在 ~-1.5 rad/s 形成
            # 稳态平衡. 若加位置项 P, 与该偏置形成正反馈环, 实测 pitch 发散到 ±0.6 rad.
            # 因此只用 D 项, 接受残留稳态 pitch_rate (落地由 LAND 阶段吸收)。
            pitch_kd = float(self.params.flight_pitch_kd)
            attitude_torque = float(np.clip(
                -pitch_kd * float(state.pitch_rate), -3.5, 3.5,
            ))
            addresses = model_addresses(model)
            control = np.zeros(model.nu)
            for geometry in LEG_CLOSED_LOOP.values():
                act_idx = addresses.actuators[geometry.motor_joint]
                lo, hi = model.actuator_ctrlrange[act_idx]
                control[act_idx] = float(np.clip(attitude_torque, lo, hi))
            return control

        target_height = self._filtered_nominal_height(float(model.opt.timestep))
        target_h_dot = 0.0   # 期望 CoM 垂直速度,用于 motor velocity FF
        target_h_ddot = 0.0  # 期望 CoM 垂直加速度,用于动态 thrust FF
        kp_motor = self.params.kp_motor
        kd_motor = self.params.kd_motor
        trajectory = self.phase_machine.trajectory if self.phase_machine is not None else None

        if phase == JumpPhase.CROUCH and trajectory is not None:
            t = self.phase_machine.time_in_phase  # type: ignore[union-attr]
            target_height = trajectory.crouch.height(t)
            target_h_dot = trajectory.crouch.velocity(t)
            target_h_ddot = trajectory.crouch.acceleration(t)
        elif phase == JumpPhase.EXTEND and trajectory is not None:
            t = self.phase_machine.time_in_phase  # type: ignore[union-attr]
            extend_duration = float(trajectory.extend.duration)
            target_height = trajectory.extend.height(t)
            target_h_dot = trajectory.extend.velocity(t)
            target_h_ddot = trajectory.extend.acceleration(t)
            if t <= extend_duration:
                # 跟轨迹: 1.5x kp/kd 跟踪轨迹位置。不能更高: 3x kp + 50ms 内
                # 快速变化的 target → motor 振荡 ±20 N·m 把执行器顶到饱和。
                kp_motor *= 1.5
                kd_motor *= 1.5
            else:
                # 轨迹跑完但相位机还在 EXTEND (弹跳确认期/能量补推): 关闭位置 kp,
                # 只保留速度跟踪 + ff_torque (= m*(g+a)/N * dh/dθ)。
                #
                # 为什么必须关 kp: ConstantAccelerationTrajectory 在 t>duration 后
                # height 被 clip 到 h_high (常数)。但 motor θ 因为 ff_torque 持续推
                # + body 上升动量,会过冲到对应 h>h_high 的角度。kp*(θ_target-θ_current)
                # 变成负的 → PID 主动把 motor 拉回 → body vz 从 1.5 m/s 在 2ms 内
                # 掉到 0.9 (实测),起跳能量被 PID 自己浪费在 ground bounce 上。
                # ff_torque 是恒定加速度 profile 的正确推力,留它继续推就行。
                kp_motor = 0.0
                kd_motor *= 1.5
        elif phase == JumpPhase.LAND and trajectory is not None:
            # 入 LAND 瞬间生成 land 轨迹。h_target 传入当前 nominal_height
            # 而不是默认 0.142,这样落地不会强行伸腿到中位然后立刻收回。
            if trajectory.land is None:
                trajectory.setup_land(
                    h_contact=current_leg_height,
                    v_contact=float(state.base_linear_velocity[2]),
                    h_target=float(self.params.nominal_height),
                )
            t = self.phase_machine.time_in_phase  # type: ignore[union-attr]
            assert trajectory.land is not None
            target_height = trajectory.land.height(t)
            target_h_dot = trajectory.land.velocity(t)
            target_h_ddot = trajectory.land.acceleration(t)
            # LAND 用独立的绝对 PD 增益 (kp_land/kd_land), 与 STAND 解耦.
            # 软 P + 硬 D 吸收落地冲击, 不再以 STAND kp_motor 的倍数表达,
            # 避免"调 STAND 同时改 LAND"的隐式耦合.
            kp_motor = float(self.params.kp_land)
            kd_motor = float(self.params.kd_land)
        # STAND: 用 filtered nominal_height,target_h_ddot=0 (静态保持)

        control = np.zeros(model.nu)
        addresses = model_addresses(model)
        total_mass = float(np.sum(model.body_mass))
        gravity = abs(float(model.opt.gravity[2]))
        if self._motor_jacobian_step_count % self.JACOBIAN_REFRESH_PERIOD == 0:
            self._motor_jacobian_cache = {
                side: _closed_loop_leg_motor_jacobian(model, data, geometry)
                for side, geometry in LEG_CLOSED_LOOP.items()
            }
            self._height_jacobian_rows_cache = _leg_height_jacobian_rows(model, data)
        jacobian = self._motor_jacobian_cache
        assert jacobian is not None
        self._motor_jacobian_step_count += 1

        neutral_offsets = self._neutral_offsets(model, data)
        roll_level_offsets = self._roll_level_height_offsets(model, data, phase, state)
        common_height = self._slope_squat_height(target_height, roll_level_offsets, phase)

        for side, geometry in LEG_CLOSED_LOOP.items():
            act_idx = addresses.actuators[geometry.motor_joint]
            motor_qpos_idx = addresses.joint_qpos[geometry.motor_joint]
            motor_qvel_idx = addresses.joint_qvel[geometry.motor_joint]

            side_target_height = common_height + neutral_offsets[side] + roll_level_offsets[side]
            self.last_target_heights[side] = side_target_height
            target_motor_angle = self.params.lut.motor_angle_from_height(side_target_height)
            previous_angle = self._target_motor_angle_prev.get(side)
            if previous_angle is None:
                self.last_target_motor_rate[side] = 0.0
            else:
                self.last_target_motor_rate[side] = (target_motor_angle - previous_angle) / float(model.opt.timestep)
            self._target_motor_angle_prev[side] = target_motor_angle

            theta_current = float(data.qpos[motor_qpos_idx])
            theta_rate = float(data.qvel[motor_qvel_idx])
            # 速度前馈: 把 trajectory 的期望 ḣ 映射到 motor 角速度,
            #   target_motor_rate = ḣ_target / (dh/dθ).
            # 没有 FF 的话 kd_motor 项 (-kd * θ_rate) 会刹住任何 motor 运动,
            # 与 trajectory 的速度需求相对抗,EXTEND 推不动。
            jacobian_side = jacobian[side]
            if abs(jacobian_side) > LEG_JACOBIAN_SINGULAR_VALUE_EPS:
                target_motor_rate = target_h_dot / jacobian_side
            else:
                target_motor_rate = 0.0
            pid_torque = (
                kp_motor * (target_motor_angle - theta_current)
                + kd_motor * (target_motor_rate - theta_rate)
            )

            if phase == JumpPhase.FLIGHT:
                # 不会到达 (FLIGHT 已在函数顶部早返回),此分支只为防御性兜底。
                ff_torque = 0.0
            else:
                # 动态前馈: 跟踪轨迹的期望加速度 ḧ_target。
                #   总 thrust = m_total * (g + ḧ_target)
                #   每腿 τ_ff = (dh/dθ) * thrust / N_legs
                # ḧ_target=0 时退化为标准重力补偿 (m*g*dh/dθ/N)。
                # 始终按物理量计算, 不留 scale 旋钮 — 任何偏离 1× 都会让稳态高度
                # 漂移, 必须靠 kp_motor 反推平衡, 把 PD 调参面变成 PD×FF 二维.
                ff_torque = (
                    jacobian[side]
                    * total_mass
                    * (gravity + target_h_ddot)
                    / len(LEG_CLOSED_LOOP)
                )

            control[act_idx] = ff_torque + pid_torque

        clipped = np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        if phase in (JumpPhase.STAND, JumpPhase.CROUCH, JumpPhase.LAND):
            for geometry in LEG_CLOSED_LOOP.values():
                act_idx = addresses.actuators[geometry.motor_joint]
                clipped[act_idx] = float(np.clip(clipped[act_idx], -3.5, 3.5))
        # EXTEND: 不做 ±3.5 二次 clip,允许动态 FF 拉满到 actuator ±12.5 推起跳。
        # FLIGHT: 已在函数顶部早返回,不会进入此分支。
        if not np.all(np.isfinite(clipped)):
            raise ValueError("VMC control must be finite")
        return clipped

    def _filtered_nominal_height(self, dt: float) -> float:
        target = float(self.params.nominal_height)
        if self._height_filtered is None:
            self._height_filtered = target
            return target
        max_step = max(float(self.params.max_height_rate), 0.0) * dt
        delta = float(np.clip(target - self._height_filtered, -max_step, max_step))
        self._height_filtered += delta
        return self._height_filtered

    def _roll_level_height_offsets(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        phase: JumpPhase,
        state: SimState,
    ) -> dict[str, float]:
        if phase != JumpPhase.STAND:
            return {side: 0.0 for side in LEG_CLOSED_LOOP}
        limit = max(float(self.params.roll_level_offset_limit), 0.0)
        if limit <= 0.0:
            return {side: 0.0 for side in LEG_CLOSED_LOOP}
        left_wheel_z = float(data.xipos[body_id(model, LEG_CLOSED_LOOP["left"].wheel_body), 2])
        right_wheel_z = float(data.xipos[body_id(model, LEG_CLOSED_LOOP["right"].wheel_body), 2])
        terrain_offset = -0.5 * (left_wheel_z - right_wheel_z)
        offset = terrain_offset + (
            float(self.params.roll_level_kp_height) * float(state.roll)
            + float(self.params.roll_level_kd_height) * float(state.roll_rate)
        )
        offset = float(np.clip(offset, -limit, limit))
        return {"left": offset, "right": -offset}

    def _slope_squat_height(
        self,
        target_height: float,
        roll_level_offsets: dict[str, float],
        phase: JumpPhase,
    ) -> float:
        """上坡降站高: 找平指令大高差时降低共模站高, 让"伸长"腿不顶 LUT 上限。

        cap = h_max - margin - 最大单侧偏置。纯前馈 (由 roll_level 偏置驱动),
        不引入 roll 闭环。平地 (偏置≈0) 时 cap > nominal → 原样返回, 不影响平地;
        非 STAND (跳跃) 时直接返回 target_height。
        """
        margin = float(self.params.slope_squat_margin)
        if margin <= 0.0 or phase != JumpPhase.STAND:
            return target_height
        max_offset = max((abs(v) for v in roll_level_offsets.values()), default=0.0)
        cap = float(self.params.lut.h_max) - margin - max_offset
        return float(min(target_height, cap))

    def _neutral_offsets(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        offsets = self._neutral_height_offsets.get(id(model))
        if offsets is None:
            average_height = _average_leg_height(model, data)
            offsets = {
                side: _leg_height(model, data, geometry.wheel_body) - average_height
                for side, geometry in LEG_CLOSED_LOOP.items()
            }
            self._neutral_height_offsets[id(model)] = offsets
        return offsets


def _average_leg_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    heights = [
        _leg_height(model, data, geometry.wheel_body)
        for geometry in LEG_CLOSED_LOOP.values()
    ]
    return float(np.mean(heights))


def _leg_height(model: mujoco.MjModel, data: mujoco.MjData, wheel_body: str) -> float:
    base_id = body_id(model, BASE_BODY_NAME)
    wheel_id = body_id(model, wheel_body)
    return float(data.xipos[base_id, 2] - data.xipos[wheel_id, 2])


def _closed_loop_leg_motor_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geometry: LegClosedLoopGeometry,
) -> float:
    addresses = model_addresses(model)
    active_dof = addresses.joint_qvel[geometry.motor_joint]
    passive_dofs = [addresses.joint_qvel[name] for name in geometry.passive_joints]

    constraint_jacobian = _connect_constraint_jacobian(model, data, geometry.equality_name)
    active_constraint = constraint_jacobian[:, active_dof : active_dof + 1]
    passive_constraint = constraint_jacobian[:, passive_dofs]

    u, singular_values, vt = np.linalg.svd(passive_constraint, full_matrices=False)
    if singular_values.size == 0 or singular_values[-1] < LEG_JACOBIAN_SINGULAR_VALUE_EPS:
        sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
        raise ValueError(f"closed-loop jacobian singular at {geometry.motor_joint} (sigma_min={sigma_min:.2e})")

    passive_constraint_pinv = vt.T @ np.diag(1.0 / singular_values) @ u.T
    passive_per_active = -passive_constraint_pinv @ active_constraint
    height_jacobian = _leg_height_jacobian_row(model, data, geometry.wheel_body)
    jacobian = float(height_jacobian[active_dof] + height_jacobian[passive_dofs] @ passive_per_active[:, 0])
    if not np.isfinite(jacobian):
        raise ValueError("VMC closed-loop leg motor jacobian must be finite")
    return jacobian


def _connect_constraint_jacobian(model: mujoco.MjModel, data: mujoco.MjData, equality_name: str) -> np.ndarray:
    constraint_id = equality_id(model, equality_name)

    body_a_id = int(model.eq_obj1id[constraint_id])
    body_b_id = int(model.eq_obj2id[constraint_id])
    if body_a_id < 0 or body_b_id < 0:
        raise ValueError(f"equality constraint must connect two bodies: {equality_name}")

    anchor_a = np.asarray(model.eq_data[constraint_id, 0:3], dtype=float)
    anchor_b = np.asarray(model.eq_data[constraint_id, 3:6], dtype=float)
    point_a = _body_local_point_world(data, body_a_id, anchor_a)
    point_b = _body_local_point_world(data, body_b_id, anchor_b)

    jacobian_a = np.zeros((3, model.nv))
    jacobian_b = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacobian_a, None, point_a, body_a_id)
    mujoco.mj_jac(model, data, jacobian_b, None, point_b, body_b_id)
    constraint_jacobian = jacobian_a - jacobian_b
    if not np.all(np.isfinite(constraint_jacobian)):
        raise ValueError("connect constraint jacobian must be finite")
    return constraint_jacobian


def _body_local_point_world(data: mujoco.MjData, body_id_value: int, local_point: np.ndarray) -> np.ndarray:
    body_rotation = data.xmat[body_id_value].reshape(3, 3)
    return data.xpos[body_id_value] + body_rotation @ local_point


def _leg_height_jacobian_rows(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    return {
        side: _leg_height_jacobian_row(model, data, geometry.wheel_body)
        for side, geometry in LEG_CLOSED_LOOP.items()
    }


def _leg_height_jacobian_row(model: mujoco.MjModel, data: mujoco.MjData, wheel_body: str) -> np.ndarray:
    base_id = body_id(model, BASE_BODY_NAME)
    wheel_id = body_id(model, wheel_body)
    base_jac = np.zeros((3, model.nv))
    wheel_jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, base_jac, None, data.xipos[base_id], base_id)
    mujoco.mj_jac(model, data, wheel_jac, None, data.xipos[wheel_id], wheel_id)
    jacobian = base_jac[2] - wheel_jac[2]
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("VMC leg height jacobian must be finite")
    return jacobian


def _leg_height_velocity_from_row(data: mujoco.MjData, jacobian: np.ndarray) -> float:
    velocity = jacobian @ data.qvel
    if not np.isfinite(velocity):
        raise ValueError("VMC leg height velocity must be finite")
    return float(velocity)
