from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.controllers.balance_lqr import LEG_ROLL_DIFF_SIGNS, equilibrium_pitch_from_geometry
from src.controllers.balance_state import balance_tangent_state_5d
from src.controllers.lqr import LqrController
from src.controllers.phase import JumpPhaseMachine, JumpPhase
from src.controllers.vmc import LEG_CLOSED_LOOP, VmcController, VmcParams
from src.model_semantics import MODEL_SEMANTICS, WHEEL_FORWARD_SIGNS
from src.state import SimState, body_id, model_addresses


JUMP_PHASES = (JumpPhase.CROUCH, JumpPhase.EXTEND, JumpPhase.FLIGHT, JumpPhase.LAND)


def _wrap_to_pi(angle: float) -> float:
    """把角度规整到 (-pi, pi], 用于航向误差的最短路径回正。"""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class CombinedParams:
    """LQR+VMC 组合控制器参数。

    架构: 相位独占 (phase-authoritative)。LQR 和 VMC 在腿 motor 上的冲突
    (height PD + roll_diff 求和饱和) 通过相位限制 LQR 各通道的输出消除。

    - STAND: LQR 全开 (轮 forward + 腿 roll 差分 + yaw) + VMC 高度 PD。
    - CROUCH/EXTEND/LAND: LQR 轮 forward + yaw 维持本体平衡 (不动腿),
      VMC 轨迹独占腿。LQR 轮 forward 阻止 pitch 在 EXTEND 期间因水平失稳放大,
      但因为不写腿 motor,leg actuator 量程完全留给 VMC 动态 FF。
    - FLIGHT: 全 0。腿 motor = 0 → 空中无 reaction torque,base 姿态由角动量
      保持;轮 = 0 → 不积累轮速,避免落地反扭。
    - FALLEN: 全 0。

    调参顺序 (实机/headless 通用, 详见 default_params.py 顶部说明):
      VMC PD → LQR balance → forward (pitch_lean) → yaw → heading_hold → jump.

    Attributes:
        vmc: VMC 高度控制参数。
        q_diag: LQR 状态权重对角线 [pitch, pitch_rate, roll, roll_rate, wheel_vel]。
            注意:Q[wheel_vel] 不应过大,否则 LQR 与外环 pitch_lean PI 争抢速度
            跟踪权限 (见 default_params.py STAND_PARAMS 注释)。
        r_diag: LQR 虚拟控制权重对角线 [forward_wheel, roll_diff_leg]。
        target_velocity: 目标前进速度 (m/s)。可运行时修改。
        pitch_lean_gain: 速度误差到目标 pitch 偏移的增益 (P)。
        velocity_ki: 速度积分补偿增益 (I)。
        position_kp: 位置外环 P 增益 (1/s)。pos_err [m] → vel_ref 修正 [m/s]。
            仅 target_velocity≈0 时启用。
        position_kd: 位置外环 D 增益。base 水平速度反馈进 vel_ref,
            提供位置-pitch 二阶环的阻尼。
        position_velocity_limit: 位置外环输出 vel_ref 修正幅度限制 (m/s)。
        yaw_damping: yaw 角速度比例阻尼增益 (P)。
        yaw_ki: yaw 角速度积分增益 (I)。
        target_yaw_rate: 目标 yaw 角速度 (rad/s)。
        heading_hold_kp: 航向保持外环 P 增益。航向误差 [rad] → yaw-rate 参考 [rad/s]。
            仅 target_yaw_rate≈0 (未发转向指令) 且站立接地时启用。与位置外环
            (_position_outer_loop) 同构: 锁定当前航向, 把航向误差串级成 yaw-rate
            参考喂进 yaw 阻尼内环, 抵抗外部扰动保持方向不变。0 = 关闭 (默认, 保持旧行为)。
        heading_hold_rate_limit: 航向保持外环输出 yaw-rate 参考的幅度上限 (rad/s),
            防止大航向误差时猛回正干扰平衡。
        fixed_height: True 时 LQR gain 锁定在 _initialize_lqr 那一刻的值,
            后续不再随几何变化重算 (适合 cmd_height 固定的场景, 节省 CPU)。
            False (默认) 动态更新 LQR gain: 静止平地高度变化时沿 height
            轴线性插值；移动或 roll-leveling 场景下只切换当前 (height_bin, roll_bin)
            缓存，避免混合不同地形姿态的线性化结果。
        lqr_height_bin_size: gain 缓存的 height bin 宽度 (m)。同一 bin 内插值
            参数确定。默认 0.02 m, 对应 LUT 操作范围内 ~5 个 bin。
        ff_gain: 高度变化引起的 wheel velocity FF 系数 (默认 0 = 禁用)。
            仅 cmd_height 快速阶跃 pitch peak 不达标时启用; 启用后又把
            height_rate 信号注入 LQR target[4], 跟 pitch_lean 路径有轻微耦合。
    """
    vmc: VmcParams
    q_diag: np.ndarray
    r_diag: np.ndarray
    target_velocity: float = 0.0
    pitch_lean_gain: float = 0.02
    velocity_ki: float = 0.1
    position_kp: float = 1.5
    position_kd: float = 1.5
    position_velocity_limit: float = 0.3
    yaw_damping: float = 0.5
    yaw_ki: float = 0.0
    target_yaw_rate: float = 0.0
    # 航向保持: 默认关闭 (kp=0), 保持旧 yaw-rate 阻尼行为不变。见 Attributes 说明。
    heading_hold_kp: float = 0.0
    heading_hold_rate_limit: float = 1.0
    fixed_height: bool = True
    lqr_height_bin_size: float = 0.02
    ff_gain: float = 0.0


class CombinedController:
    """相位独占控制器: STAND 用 LQR+VMC,跳跃全程 VMC 独占。"""

    def __init__(self, params: CombinedParams, phase_machine: JumpPhaseMachine | None = None) -> None:
        self.params = params
        self.vmc_controller = VmcController(params.vmc, phase_machine)
        self._lqr_controller: LqrController | None = None
        self._velocity_integral: float = 0.0
        self._yaw_integral: float = 0.0
        # Gain 表键 = (height_bin, roll_bin)。每个 roll_bin 第一次出现时一次性填满
        # height 轴，避免 cmd_height 扫描时在控制循环里反复求解 DARE。
        self._lqr_gain_cache: dict[tuple[int, int], np.ndarray] = {}
        self._wheel_ff_gain_cache: dict[tuple[int, int], float] = {}
        self._prewarmed_roll_bins: set[int] = set()
        self._active_wheel_ff_gain: float = 0.0
        self._height_wheel_velocity_ff: float = 0.0
        self._last_phase = JumpPhase.STAND
        self._position_anchor: np.ndarray | None = None
        self._heading_anchor: float | None = None
        self._equilibrium_pitch: float = 0.0

    @property
    def lqr_controller(self) -> LqrController | None:
        return self._lqr_controller

    # ---------- LQR gain management ----------

    def _bin_size(self) -> float:
        return max(float(self.params.lqr_height_bin_size), 1e-6)

    def _roll_bin(self, state: SimState) -> int:
        return int(np.floor(float(state.roll) / 0.02))

    def _height_bin(self, state: SimState) -> int:
        return int(np.floor(float(state.base_position[2]) / self._bin_size()))

    def _height_table_bins(self) -> range:
        lut = self.params.vmc.lut
        bin_size = self._bin_size()
        lo = int(np.floor(float(lut.h_min) / bin_size)) - 2
        hi = int(np.ceil(float(lut.h_max) / bin_size)) + 2
        return range(lo, hi + 1)

    def _prewarm_lqr_height_table(self, model: Any, data: Any, h_bin: int, r_bin: int) -> None:
        if r_bin in self._prewarmed_roll_bins:
            return
        gain = self._gain_for_bin(model, data, h_bin, r_bin)
        ff = self._wheel_ff_for_bin(model, data, h_bin, r_bin)
        for table_h_bin in self._height_table_bins():
            self._lqr_gain_cache.setdefault((table_h_bin, r_bin), gain)
            self._wheel_ff_gain_cache.setdefault((table_h_bin, r_bin), ff)
        self._prewarmed_roll_bins.add(r_bin)

    def _gain_for_bin(
        self, model: Any, data: Any, h_bin: int, r_bin: int,
    ) -> np.ndarray:
        cached = self._lqr_gain_cache.get((h_bin, r_bin))
        if cached is not None:
            return cached
        from src.controllers.balance_lqr import compute_balance_lqr_gain_5d
        gain = compute_balance_lqr_gain_5d(
            model, data, self.params.q_diag, self.params.r_diag,
        )
        self._lqr_gain_cache[(h_bin, r_bin)] = gain
        return gain

    def _wheel_ff_for_bin(
        self, model: Any, data: Any, h_bin: int, r_bin: int,
    ) -> float:
        cached = self._wheel_ff_gain_cache.get((h_bin, r_bin))
        if cached is not None:
            return cached
        from src.controllers.balance_lqr import wheel_ff_gain_for_leg_common
        ff = wheel_ff_gain_for_leg_common(model, data)
        self._wheel_ff_gain_cache[(h_bin, r_bin)] = ff
        return ff

    def _interpolated_lqr_inputs(
        self, model: Any, data: Any, state: SimState,
    ) -> tuple[np.ndarray, float]:
        h_bin = self._height_bin(state)
        r_bin = self._roll_bin(state)
        if not self._should_interpolate_height_gain(state):
            return (
                self._gain_for_bin(model, data, h_bin, r_bin),
                self._wheel_ff_for_bin(model, data, h_bin, r_bin),
            )

        self._prewarm_lqr_height_table(model, data, h_bin, r_bin)
        h = float(state.base_position[2])
        center_position = h / self._bin_size() - 0.5
        h_lo = int(np.floor(center_position))
        h_hi = h_lo + 1
        alpha = float(np.clip(center_position - h_lo, 0.0, 1.0))

        gain_lo = self._gain_for_bin(model, data, h_lo, r_bin)
        gain_hi = self._gain_for_bin(model, data, h_hi, r_bin)
        ff_lo = self._wheel_ff_for_bin(model, data, h_lo, r_bin)
        ff_hi = self._wheel_ff_for_bin(model, data, h_hi, r_bin)
        return (
            (1.0 - alpha) * gain_lo + alpha * gain_hi,
            (1.0 - alpha) * ff_lo + alpha * ff_hi,
        )

    def _should_interpolate_height_gain(self, state: SimState) -> bool:
        if self.params.fixed_height:
            return False
        if abs(float(self.params.target_velocity)) > 1e-6:
            return False
        if abs(float(state.roll)) >= 0.015 or abs(float(state.roll_rate)) >= 0.2:
            return False
        vmc = self.params.vmc
        return not (
            abs(float(vmc.roll_level_kp_height)) > 1e-12
            or abs(float(vmc.roll_level_kd_height)) > 1e-12
            or abs(float(vmc.roll_level_offset_limit)) > 1e-12
        )

    def _initialize_lqr(self, model: Any, data: Any, state: SimState) -> None:
        if not np.isfinite(self.params.target_velocity):
            raise ValueError("target_velocity must be finite")
        gain, ff_gain = self._interpolated_lqr_inputs(model, data, state)
        self._active_wheel_ff_gain = ff_gain
        self._equilibrium_pitch = equilibrium_pitch_from_geometry(model, data)
        target = balance_tangent_state_5d(model, data, state).copy()
        target[0] = self._equilibrium_pitch
        target[1] = 0.0
        target[2] = 0.0
        target[3] = 0.0
        target[4] = self.params.target_velocity
        self._lqr_controller = LqrController(gain, target, np.zeros(2), balance_tangent_state_5d)

    def _ensure_lqr_height_bin(self, model: Any, data: Any, state: SimState) -> None:
        if self._lqr_controller is None:
            return
        gain, ff_gain = self._interpolated_lqr_inputs(model, data, state)
        self._lqr_controller.gain = gain
        self._active_wheel_ff_gain = ff_gain

    # ---------- Height feedforward ----------

    def _average_leg_height_and_wheel_mid_z(self, model: Any, data: Any) -> tuple[float, float]:
        wheel_ids = [body_id(model, geometry.wheel_body) for geometry in LEG_CLOSED_LOOP.values()]
        base_id = body_id(model, "base_link")
        wheel_mid_z = float(np.mean([data.xipos[wheel_id, 2] for wheel_id in wheel_ids]))
        return float(data.xipos[base_id, 2] - wheel_mid_z), wheel_mid_z

    def _height_wheel_velocity_feedforward(self, current_leg_height: float) -> float:
        if self.params.fixed_height:
            return 0.0
        left_rate = self.vmc_controller.last_target_motor_rate.get("left", 0.0)
        right_rate = self.vmc_controller.last_target_motor_rate.get("right", 0.0)
        theta_rate_mean = 0.5 * (left_rate + right_rate)
        dh_dt_cmd = self.params.vmc.lut.height_dtheta(current_leg_height) * theta_rate_mean
        return self.params.ff_gain * self.params.vmc.lut.dy_wheel_dh(current_leg_height) * dh_dt_cmd

    # ---------- Position / velocity outer loops ----------

    def _position_outer_loop(self, model: Any, data: Any, state: SimState) -> float:
        """位置 PD 外环。target_velocity=0 且有 anchor 时启用。仅在 STAND 调用。"""
        if self._position_anchor is None:
            return 0.0
        if abs(self.params.target_velocity) > 1e-6:
            return 0.0
        base_id = body_id(model, "base_link")
        rotation = np.asarray(data.xmat[base_id]).reshape(3, 3)
        forward_horiz = rotation[:2, 1]
        forward_norm = float(np.linalg.norm(forward_horiz))
        if forward_norm < 1e-6:
            return 0.0
        forward_horiz = forward_horiz / forward_norm
        delta = self._position_anchor - state.base_position[:2]
        forward_error = float(np.dot(delta, forward_horiz))
        forward_velocity = float(np.dot(state.base_linear_velocity[:2], forward_horiz))
        vel_correction = self.params.position_kp * forward_error - self.params.position_kd * forward_velocity
        limit = max(float(self.params.position_velocity_limit), 0.0)
        return float(np.clip(vel_correction, -limit, limit))

    def _base_heading(self, model: Any, data: Any) -> float:
        """本体前向 (+Y) 在世界 XY 平面投影的航向角 (rad)。

        与 _position_outer_loop 的 forward 定义一致 (rotation[:,1] = 本体 Y 轴)。
        投影到水平面后取 atan2, 对俯仰倾角不敏感 (pitch 只改变投影长度不改方位角)。
        """
        base_id = body_id(model, "base_link")
        rotation = np.asarray(data.xmat[base_id]).reshape(3, 3)
        forward_horiz = rotation[:2, 1]
        if float(np.linalg.norm(forward_horiz)) < 1e-6:
            return 0.0
        return float(np.arctan2(forward_horiz[1], forward_horiz[0]))

    def _heading_outer_loop(self, model: Any, data: Any) -> float:
        """航向保持 P 外环。返回 yaw-rate 参考 (rad/s), 串级进 _compute_yaw_correction。

        仅在: 有 anchor + 未发转向指令 (target_yaw_rate≈0) + heading_hold_kp>0 时输出。
        error = wrap(anchor - heading); rate_ref = clip(kp*error, ±rate_limit)。
        sign: 实际 yaw_rate>0 (本体绕 +Z, 上升航向角) 会减小 error, 故 rate_ref 与
        error 同号即构成稳定回正 (内环把 yaw_rate 拉向 rate_ref)。仅 STAND 接地时调用。
        """
        if self._heading_anchor is None:
            return 0.0
        if abs(self.params.target_yaw_rate) > 1e-6:
            return 0.0
        kp = float(self.params.heading_hold_kp)
        if kp <= 0.0:
            return 0.0
        error = _wrap_to_pi(self._heading_anchor - self._base_heading(model, data))
        limit = max(float(self.params.heading_hold_rate_limit), 0.0)
        return float(np.clip(kp * error, -limit, limit))

    def _update_lqr_target(self, model: Any, data: Any, state: SimState, dt: float) -> None:
        """STAND 阶段的 LQR 目标更新。equilibrium_pitch 跟随当前几何重算。

        5D state: [pitch, pitch_rate, roll, roll_rate, wheel_vel]。
        target[0] = equilibrium_pitch + pitch_lean (速度 PI 输出);
        target[4] = target_velocity + position_outer_loop (位置 P 输出) + height_ff。
        """
        if self._lqr_controller is None:
            return
        current_tangent = balance_tangent_state_5d(None, None, state)
        current_wheel_vel = float(current_tangent[4])

        self._equilibrium_pitch = equilibrium_pitch_from_geometry(model, data)
        position_vel_correction = self._position_outer_loop(model, data, state)

        velocity_target = self.params.target_velocity + self._height_wheel_velocity_ff + position_vel_correction
        velocity_error = velocity_target - current_wheel_vel

        # Anti-windup: 积分只在 pitch_lean 未饱和时累积。
        max_lean = 0.2
        pitch_p = self.params.pitch_lean_gain * velocity_error
        pitch_i = self.params.velocity_ki * self._velocity_integral
        pitch_lean = pitch_p + pitch_i
        if -max_lean < pitch_lean < max_lean:
            self._velocity_integral += velocity_error * dt
        pitch_lean = float(np.clip(pitch_lean, -max_lean, max_lean))

        self._lqr_controller.target[0] = self._equilibrium_pitch + pitch_lean
        self._lqr_controller.target[1] = 0.0
        self._lqr_controller.target[2] = 0.0
        self._lqr_controller.target[3] = 0.0
        self._lqr_controller.target[4] = velocity_target

    def _compute_yaw_correction(self, state: SimState, dt: float, heading_rate_ref: float = 0.0) -> float:
        """yaw 角速度阻尼内环。effective_target = target_yaw_rate + 航向保持外环参考。

        航向保持把航向角误差转成 yaw-rate 参考 (heading_rate_ref) 串级进来; 不发转向
        指令时它驱动本环把实际 yaw_rate 拉向"回正所需角速度", 从而把航向拉回 anchor。
        heading_rate_ref=0 时退化为原始 yaw-rate 阻尼 (跳跃相位即走此路径)。
        """
        yaw_rate = float(state.base_angular_velocity[2])
        effective_target_yaw_rate = self.params.target_yaw_rate + heading_rate_ref
        yaw_error = yaw_rate - effective_target_yaw_rate
        self._yaw_integral += yaw_error * dt
        self._yaw_integral = float(np.clip(self._yaw_integral, -5.0, 5.0))
        return self.params.yaw_damping * yaw_error + self.params.yaw_ki * self._yaw_integral

    # ---------- Control allocation ----------

    def _allocate_balance_control(
        self,
        model: Any,
        forward_torque: float,
        roll_torque: float,
        yaw_torque: float,
        addresses: Any,
    ) -> np.ndarray:
        """STAND 模式: 把 LQR 虚拟力矩 + yaw 修正分配到物理轮/腿执行器。"""
        control = np.zeros(model.nu)
        virtual_wheel_torques = (forward_torque - yaw_torque, forward_torque + yaw_torque)
        for joint_name, virtual_torque in zip(MODEL_SEMANTICS.wheel_joints, virtual_wheel_torques):
            actuator_index = addresses.actuators[joint_name]
            control[actuator_index] = WHEEL_FORWARD_SIGNS[joint_name] * virtual_torque
        for joint_name, sign in LEG_ROLL_DIFF_SIGNS.items():
            control[addresses.actuators[joint_name]] = sign * roll_torque
        return control

    def _merge_vmc_and_clip(
        self,
        model: Any,
        control: np.ndarray,
        vmc_control: np.ndarray,
        addresses: Any,
        phase: JumpPhase,
    ) -> np.ndarray:
        """合并 VMC 腿力矩 (共模高度控制) 到 control,然后按相位 clip。

        只 clip 物理执行器 (轮 + 腿 motor),不动 cmd_* slider 这些非物理 actuator
        (它们由 launch_mujoco 在 step 前后单独 read/restore)。
        """
        for joint_name in MODEL_SEMANTICS.leg_motor_joints:
            actuator_index = addresses.actuators[joint_name]
            control[actuator_index] += vmc_control[actuator_index]

        clipped = control.copy()
        physical_joints = MODEL_SEMANTICS.wheel_joints + MODEL_SEMANTICS.leg_motor_joints
        for joint_name in physical_joints:
            act_idx = addresses.actuators[joint_name]
            clipped[act_idx] = float(np.clip(
                control[act_idx],
                model.actuator_ctrlrange[act_idx, 0],
                model.actuator_ctrlrange[act_idx, 1],
            ))
        if phase in (JumpPhase.STAND, JumpPhase.CROUCH, JumpPhase.LAND):
            for joint_name in MODEL_SEMANTICS.leg_motor_joints:
                act_idx = addresses.actuators[joint_name]
                clipped[act_idx] = float(np.clip(clipped[act_idx], -3.5, 3.5))
        # EXTEND: 允许动态 FF 拉满到 actuator 极限 (±12.5 N·m) 推起跳。
        # FLIGHT: VMC 早返回 0,本路径在 _jump_control 下也只会得到 0 + 0 = 0。

        if not np.all(np.isfinite(clipped)):
            raise ValueError("combined control must be finite")
        return clipped

    # ---------- State management ----------

    def _reset_balance_state(self) -> None:
        self._velocity_integral = 0.0
        self._yaw_integral = 0.0
        self._height_wheel_velocity_ff = 0.0
        self._lqr_controller = None
        self._position_anchor = None
        self._heading_anchor = None

    def _handle_stand_entry(self, model: Any, data: Any, state: SimState) -> None:
        """STAND 进入瞬间: 清积分,锁位置 anchor + 航向 anchor。"""
        if self._last_phase != JumpPhase.STAND:
            self._velocity_integral = 0.0
            self._position_anchor = np.array(state.base_position[:2], dtype=float)
            self._heading_anchor = self._base_heading(model, data)
        if abs(self.params.target_velocity) > 1e-6:
            self._position_anchor = None
        elif self._position_anchor is None:
            self._position_anchor = np.array(state.base_position[:2], dtype=float)
        # 航向 anchor 与位置 anchor 独立: 发转向指令时丢弃, 松开 (≈0) 时重新锁定当前航向。
        # 注意只看 target_yaw_rate, 与 target_velocity 无关 — 直线行驶 (有速度无转向)
        # 时仍保持航向, 抵抗偏航漂移。
        if abs(self.params.target_yaw_rate) > 1e-6:
            self._heading_anchor = None
        elif self._heading_anchor is None:
            self._heading_anchor = self._base_heading(model, data)

    def _handle_jump_entry(self) -> None:
        """跳跃序列开始: 清积分,丢 anchor。LQR controller 本身保留 (gain 只依赖几何)。"""
        if self._last_phase not in JUMP_PHASES:
            self._velocity_integral = 0.0
            self._yaw_integral = 0.0
            self._position_anchor = None
            self._heading_anchor = None
        self._height_wheel_velocity_ff = 0.0

    # ---------- Phase-dispatched control ----------

    def _stand_control(
        self,
        model: Any,
        data: Any,
        state: SimState,
        dt: float,
        vmc_control: np.ndarray,
        addresses: Any,
    ) -> np.ndarray:
        """STAND: LQR 全开 (轮 forward + 腿 roll 差分 + yaw) + VMC 高度 PD。"""
        if self._lqr_controller is None:
            self._initialize_lqr(model, data, state)
        if self._lqr_controller is None:
            raise ValueError("failed to initialize LQR controller")
        if not self.params.fixed_height:
            self._ensure_lqr_height_bin(model, data, state)

        current_leg_height, _ = self._average_leg_height_and_wheel_mid_z(model, data)
        self._height_wheel_velocity_ff = self._height_wheel_velocity_feedforward(current_leg_height)

        self._update_lqr_target(model, data, state, dt)
        # Feedforward 抵消 VMC 共模 leg torque 引起的 pitch 扰动:
        # VMC 在 leg motor 上输出 τ_L, τ_R (vmc_control), 共模 = (τ_L + τ_R)/2.
        # 这个 common torque 通过 hip motor 反作用产生 pitch_rate 扰动
        # (实测系数 ≈ -0.15 rad/s² per N·m·leg, 见 _leg_common_to_pitch_coupling).
        # 用 wheel 提前抵消, LQR 反馈只处理残差。
        left_leg, right_leg = MODEL_SEMANTICS.leg_motor_joints
        vmc_common_leg = 0.5 * (
            float(vmc_control[addresses.actuators[left_leg]])
            + float(vmc_control[addresses.actuators[right_leg]])
        )
        wheel_ff = self._active_wheel_ff_gain * vmc_common_leg
        self._lqr_controller.feedforward = np.array([wheel_ff, 0.0])
        lqr_control = self._lqr_controller(model, data, state)
        if lqr_control.shape != (2,):
            raise ValueError("LQR virtual control must have shape (2,)")
        if state.contact_count >= len(LEG_CLOSED_LOOP):
            forward_torque = float(lqr_control[0])
            heading_rate_ref = self._heading_outer_loop(model, data)
            yaw_correction = self._compute_yaw_correction(state, dt, heading_rate_ref)
        else:
            forward_torque = 0.0
            yaw_correction = 0.0
        # Roll leveling is handled by VMC as differential leg-height targets.
        # Do not also write LQR roll torque into the same leg actuators.
        roll_torque = 0.0

        control = self._allocate_balance_control(model, forward_torque, roll_torque, yaw_correction, addresses)
        return self._merge_vmc_and_clip(model, control, vmc_control, addresses, JumpPhase.STAND)

    def _set_lqr_target_balance_only(self, model: Any, data: Any) -> None:
        """跳跃期间的 LQR 目标: 纯 equilibrium_pitch,wheel_vel=0,无 pitch_lean。

        STAND 的 _update_lqr_target 包含 pitch_lean (从 velocity_error 推出),
        用于跟踪 target_velocity > 0 时身体前倾。但跳跃期间 wheel velocity 会被
        起跳动力学瞬时拉到几 rad/s,pitch_lean 把这个误差当成"该前倾",saturate
        到 ±0.2 rad,反而强行让 LQR 把车体推倒。跳跃只需要保持竖直 (target=0)。
        """
        if self._lqr_controller is None:
            return
        equilibrium_pitch = equilibrium_pitch_from_geometry(model, data)
        self._lqr_controller.target[0] = equilibrium_pitch
        self._lqr_controller.target[1] = 0.0
        self._lqr_controller.target[2] = 0.0
        self._lqr_controller.target[3] = 0.0
        self._lqr_controller.target[4] = 0.0  # 跳跃期间不追速度,只保持原地

    def _balance_only_control(
        self,
        model: Any,
        data: Any,
        state: SimState,
        dt: float,
        vmc_control: np.ndarray,
        addresses: Any,
        phase: JumpPhase,
    ) -> np.ndarray:
        """CROUCH/EXTEND/LAND: LQR 轮 forward + yaw 维持平衡,但 NOT 写腿 (VMC 独占)。

        LQR target: 纯 equilibrium_pitch + wheel_vel=0,不带 pitch_lean。跳跃中
        wheel velocity 被动力学拉到几 rad/s,pitch_lean 会把这视为"应该前倾"
        从而把车推倒 — 用 _set_lqr_target_balance_only 隔离这个机制。

        EXTEND 期间 LQR 轮 forward 关键: 自由发展的 pitch 会在 70-150ms 内长到
        ~0.1 rad (sqrt(g/L) 不稳定模态),让 thrust 大量分解到水平,起跳不成功;
        LQR 抑制 pitch 漂移。腿 motor 输出由 VMC 独占。

        airborne 安全: contact_count == 0 时 LQR forward 和 yaw 清零。无地面
        摩擦时任何 wheel 力矩只会让 wheel 自由加速,重新接地切向滑动反扭翻车。
        """
        if self._lqr_controller is None:
            self._initialize_lqr(model, data, state)
        if self._lqr_controller is None:
            raise ValueError("failed to initialize LQR controller")
        if not self.params.fixed_height:
            self._ensure_lqr_height_bin(model, data, state)

        self._set_lqr_target_balance_only(model, data)
        self._lqr_controller.feedforward = np.zeros(2)
        lqr_control = self._lqr_controller(model, data, state)
        if state.contact_count == 0:
            forward_torque = 0.0
            yaw_correction = 0.0
        else:
            forward_torque = float(lqr_control[0])
            yaw_correction = self._compute_yaw_correction(state, dt)
        # roll_diff_leg = 0: 不写腿 motor,把整个 leg actuator 量程让给 VMC。

        control = self._allocate_balance_control(model, forward_torque, 0.0, yaw_correction, addresses)
        return self._merge_vmc_and_clip(model, control, vmc_control, addresses, phase)

    def __call__(self, model: Any, data: Any, state: SimState) -> np.ndarray:
        dt = float(model.opt.timestep)
        addresses = model_addresses(model)
        vmc_control = self.vmc_controller(model, data, state)

        phase = (
            self.vmc_controller.phase_machine.phase
            if self.vmc_controller.phase_machine is not None
            else JumpPhase.STAND
        )

        if phase == JumpPhase.FALLEN:
            self._reset_balance_state()
            self._last_phase = phase
            return np.zeros(model.nu)

        if phase == JumpPhase.FLIGHT:
            # 不应用 LQR (无地面 wheel torque 没意义),但保留 VMC 的腿电机输出 —
            # VMC FLIGHT 分支用对称腿做 pitch_rate 反作用阻尼,防止空中翻车。
            # 见 vmc.py FLIGHT 分支。
            self._handle_jump_entry()
            self._last_phase = phase
            return vmc_control

        if phase == JumpPhase.STAND:
            self._handle_stand_entry(model, data, state)
            control = self._stand_control(model, data, state, dt, vmc_control, addresses)
        else:
            # CROUCH / EXTEND / LAND: LQR 轮平衡 + VMC 轨迹独占腿
            self._handle_jump_entry()
            control = self._balance_only_control(model, data, state, dt, vmc_control, addresses, phase)

        self._last_phase = phase
        return control
