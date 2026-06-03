"""跳跃 CoM 高度轨迹规划 (业界 4 相: CROUCH / EXTEND / FLIGHT / LAND)。

参考: SLIP + Hierarchical Jumping Optimization。每相位用 5 阶多项式连接
position/velocity/acceleration 边界条件,VMC 沿轨迹做动态前馈跟踪。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


GRAVITY = 9.81


class QuinticTrajectory:
    """5 次多项式:h(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5。

    6 个边界条件: 起点 (h0, hd0, hdd0) 和终点 (hf, hdf, hddf)。
    """

    def __init__(
        self,
        h0: float,
        hd0: float,
        hdd0: float,
        hf: float,
        hdf: float,
        hddf: float,
        duration: float,
    ) -> None:
        if duration <= 0.0:
            raise ValueError(f"QuinticTrajectory duration must be > 0, got {duration}")
        self.duration = float(duration)

        T = float(duration)
        # 起点条件直接给出 c0/c1/c2:
        self._c0 = float(h0)
        self._c1 = float(hd0)
        self._c2 = float(hdd0) / 2.0

        # 终点条件解 c3/c4/c5: 3x3 线性方程组
        # h(T)  = c0 + c1*T + c2*T^2 + c3*T^3 + c4*T^4 + c5*T^5 = hf
        # h'(T) = c1 + 2*c2*T + 3*c3*T^2 + 4*c4*T^3 + 5*c5*T^4 = hdf
        # h''(T)= 2*c2 + 6*c3*T + 12*c4*T^2 + 20*c5*T^3 = hddf
        rhs = np.array([
            hf - self._c0 - self._c1 * T - self._c2 * T**2,
            hdf - self._c1 - 2.0 * self._c2 * T,
            hddf - 2.0 * self._c2,
        ], dtype=float)
        A = np.array([
            [T**3, T**4, T**5],
            [3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
            [6.0 * T, 12.0 * T**2, 20.0 * T**3],
        ], dtype=float)
        c3, c4, c5 = np.linalg.solve(A, rhs)
        self._c3 = float(c3)
        self._c4 = float(c4)
        self._c5 = float(c5)

    def _clip_t(self, t: float) -> float:
        return float(np.clip(t, 0.0, self.duration))

    def height(self, t: float) -> float:
        t = self._clip_t(t)
        return (
            self._c0 + self._c1 * t + self._c2 * t**2
            + self._c3 * t**3 + self._c4 * t**4 + self._c5 * t**5
        )

    def velocity(self, t: float) -> float:
        t = self._clip_t(t)
        return (
            self._c1 + 2.0 * self._c2 * t + 3.0 * self._c3 * t**2
            + 4.0 * self._c4 * t**3 + 5.0 * self._c5 * t**4
        )

    def acceleration(self, t: float) -> float:
        t = self._clip_t(t)
        return (
            2.0 * self._c2 + 6.0 * self._c3 * t + 12.0 * self._c4 * t**2
            + 20.0 * self._c5 * t**3
        )

    def sample(self, t: float) -> Tuple[float, float, float]:
        return self.height(t), self.velocity(t), self.acceleration(t)


class QuarticTrajectory:
    """4 次多项式: h(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4。

    5 个边界条件: 起点 (h0, hd0, hdd0) + 终点 (hf, hdf)。
    用于 EXTEND 段 — 5 阶多项式带 hddf=0 会产生起点 "下凹" (h 先下降后上升),
    motor target 反向跟随,EXTEND 推不动。4 阶不约束 hddf,h(t) 单调递增。
    """

    def __init__(
        self,
        h0: float,
        hd0: float,
        hdd0: float,
        hf: float,
        hdf: float,
        duration: float,
    ) -> None:
        if duration <= 0.0:
            raise ValueError(f"QuarticTrajectory duration must be > 0, got {duration}")
        self.duration = float(duration)

        T = float(duration)
        self._c0 = float(h0)
        self._c1 = float(hd0)
        self._c2 = float(hdd0) / 2.0

        # 终点条件解 c3/c4: 2x2 线性方程组
        rhs = np.array([
            hf - self._c0 - self._c1 * T - self._c2 * T**2,
            hdf - self._c1 - 2.0 * self._c2 * T,
        ], dtype=float)
        A = np.array([
            [T**3, T**4],
            [3.0 * T**2, 4.0 * T**3],
        ], dtype=float)
        c3, c4 = np.linalg.solve(A, rhs)
        self._c3 = float(c3)
        self._c4 = float(c4)

    def _clip_t(self, t: float) -> float:
        return float(np.clip(t, 0.0, self.duration))

    def height(self, t: float) -> float:
        t = self._clip_t(t)
        return self._c0 + self._c1 * t + self._c2 * t**2 + self._c3 * t**3 + self._c4 * t**4

    def velocity(self, t: float) -> float:
        t = self._clip_t(t)
        return self._c1 + 2.0 * self._c2 * t + 3.0 * self._c3 * t**2 + 4.0 * self._c4 * t**3

    def acceleration(self, t: float) -> float:
        t = self._clip_t(t)
        return 2.0 * self._c2 + 6.0 * self._c3 * t + 12.0 * self._c4 * t**2

    def sample(self, t: float) -> Tuple[float, float, float]:
        return self.height(t), self.velocity(t), self.acceleration(t)


class ConstantAccelerationTrajectory:
    """恒定加速度起跳轨迹: h(t) = h0 + 0.5*a*t²,从静止加速到 v_target。

    由 stroke d = h_target - h0 和 v_target 反推 duration:
        v² = 2*a*d  →  a = v²/(2d)
        v = a*T     →  T = v/a = 2d/v
    这样电机从 t=0 就提供恒定推力 a + g (而不是 QuarticTrajectory 那种后置爆发),
    在有限的 leg stroke 内最大化能量传递。

    用于 EXTEND 段。CROUCH/LAND 用 QuinticTrajectory (需要平滑两端的速度/加速度)。
    """

    def __init__(self, h0: float, h_target: float, v_target: float) -> None:
        d = float(h_target) - float(h0)
        v = float(v_target)
        if d <= 0.0:
            raise ValueError(f"ConstantAccelerationTrajectory needs h_target > h0, got d={d}")
        if v <= 0.0:
            raise ValueError(f"ConstantAccelerationTrajectory needs v_target > 0, got v={v}")
        self.duration = 2.0 * d / v
        self._a = v / self.duration  # = v² / (2*d)
        self._h0 = float(h0)

    def _clip_t(self, t: float) -> float:
        return float(np.clip(t, 0.0, self.duration))

    def height(self, t: float) -> float:
        t = self._clip_t(t)
        return self._h0 + 0.5 * self._a * t * t

    def velocity(self, t: float) -> float:
        t = self._clip_t(t)
        return self._a * t

    def acceleration(self, t: float) -> float:
        # 恒定加速度,与 t 无关
        del t
        return self._a

    def sample(self, t: float) -> Tuple[float, float, float]:
        return self.height(t), self.velocity(t), self.acceleration(t)


@dataclass(frozen=True)
class JumpTrajectoryParams:
    """跳跃轨迹规划参数。

    EXTEND duration 由 stroke (h_high - h_low) 和 v_takeoff 反推,不再独立指定。
    """

    # LUT 安全区: h_min 是 LUT 数据下限, h_safe_high 避开奇异区 (theta_max=0.65 对应 h~0.154)。
    h_min: float = 0.0785
    # h_safe_high 现在是 EXTEND 终点的上限 (ceiling), 不再是固定终点。
    h_safe_high: float = 0.140
    # CROUCH 自适应深度: target = max(h_min, h_start - crouch_depth)。
    crouch_depth: float = 0.05
    # EXTEND 固定伸腿行程 (m): h_high = h_low + extend_stroke。固定行程让不同 cmd_height
    # 起跳的伸腿动力学一致 (离地注入机身的后仰角动量一致), 消除低 cmd_height 起跳行程
    # 过大导致的落地前倾/漂移。撞 h_safe_high 上限时整体下移窗口以保持行程 (见 JumpTrajectory)。
    extend_stroke: float = 0.045
    # 时间剖面 (EXTEND 由 ConstantAccelerationTrajectory 自动计算 duration)。
    crouch_duration: float = 0.25
    land_duration: float = 0.25
    # 默认空中高度 (m),cmd_jump=1 时跳多高。
    air_height_max: float = 0.10
    # 落地后回到 stand 的目标高度 (mid LUT)。
    h_stand_after_land: float = 0.142

    def adaptive_crouch_target(self, h_start: float) -> float:
        return max(float(self.h_min), float(h_start) - float(self.crouch_depth))


class JumpTrajectory:
    """整个跳跃过程的轨迹。

    - CROUCH: QuinticTrajectory,h_start → h_low,两端静止。
    - EXTEND: ConstantAccelerationTrajectory,h_low → h_high,从静止匀加速到 v_takeoff。
      duration 由 stroke 和 v_takeoff 反推,而非独立指定 — 保证电机从 t=0 就以恒定
      推力工作,在有限 stroke 内最大化能量传递。
    - LAND: QuinticTrajectory,着地时生成,吸收落地冲量。
    - FLIGHT: 不规划 (空中无控)。
    """

    def __init__(
        self,
        params: JumpTrajectoryParams,
        h_start: float,
        cmd_jump_amplitude: float,
    ) -> None:
        self.params = params
        self.h_start = float(h_start)
        # LAND 终点 = 起跳前的高度。如果固定用 params.h_stand_after_land (0.142),
        # 当用户 cmd_height < 0.142 时 LAND 把腿伸到中位 → STAND 立刻把腿收回,
        # 产生"落地瞬间突兀伸腿再砸下"的诡异动作。
        self.h_target_after_land = float(h_start)

        amp = float(np.clip(cmd_jump_amplitude, 0.0, 1.0))
        self.h_air = amp * float(params.air_height_max)
        # 弹道起跳速度: v_takeoff = sqrt(2 g h_air)。h_air=0 时为 0 (无跳)。
        self.v_takeoff = float(np.sqrt(2.0 * GRAVITY * max(self.h_air, 0.0)))

        h_low = params.adaptive_crouch_target(h_start)
        # 固定 EXTEND 行程: h_high = h_low + extend_stroke。若超过 h_safe_high (避开奇异区
        # 的上限) 则封顶, 并把整个起跳窗口下移 (h_low 随之下降, 不低于 h_min), 保持行程
        # 一致。这样所有 cmd_height 起跳的伸腿行程相同 → 离地注入机身的后仰角动量相同,
        # 不再出现低 cmd_height 起跳行程过大、落地前倾/漂移放大的现象。
        h_high = h_low + float(params.extend_stroke)
        if h_high > float(params.h_safe_high):
            h_high = float(params.h_safe_high)
            h_low = max(float(params.h_min), h_high - float(params.extend_stroke))

        self.h_low = h_low
        self.h_high = h_high

        # CROUCH: h_start → h_low, 两端速度/加速度 = 0
        self.crouch = QuinticTrajectory(
            h0=self.h_start, hd0=0.0, hdd0=0.0,
            hf=h_low, hdf=0.0, hddf=0.0,
            duration=float(params.crouch_duration),
        )
        # EXTEND: 恒定加速度从 h_low 到 h_high,末速度 = v_takeoff。duration 自动。
        # 0 跳跃情况 (v_takeoff=0) 用占位 QuarticTrajectory (不会实际触发起跳)。
        if self.v_takeoff > 1e-6:
            self.extend: Any = ConstantAccelerationTrajectory(
                h0=h_low, h_target=h_high, v_target=self.v_takeoff,
            )
        else:
            # cmd_jump_amplitude=0 时 v_takeoff=0,用一个不会移动的占位 trajectory
            self.extend = QuarticTrajectory(
                h0=h_low, hd0=0.0, hdd0=0.0,
                hf=h_low, hdf=0.0,
                duration=0.01,
            )
        # LAND 在着地时生成 (依赖落地速度/高度)
        self.land: QuinticTrajectory | None = None

    def setup_land(self, h_contact: float, v_contact: float, h_target: float | None = None) -> None:
        """着地瞬间生成 LAND 轨迹: h_contact → h_target (默认 = 起跳前高度)。

        v_contact 应为负 (向下),终点静止。h_target 通常传入 VMC 当前的
        nominal_height (即 cmd_height 滑条值),这样落地后的目标就是用户实际
        想要的站立姿态,避免 LAND 把腿强行伸到固定的 0.142 中位、然后 STAND
        立刻收回的诡异"突兀伸腿"动作。
        """
        target = float(h_target) if h_target is not None else float(self.h_target_after_land)
        self.land = QuinticTrajectory(
            h0=float(h_contact), hd0=float(v_contact), hdd0=0.0,
            hf=target, hdf=0.0, hddf=0.0,
            duration=float(self.params.land_duration),
        )

    # CROUCH/EXTEND/LAND 期望的弹道顶点 (供相位机判断是否完成)
    @property
    def crouch_target(self) -> float:
        return self.h_low

    @property
    def extend_target(self) -> float:
        return self.h_high

    def is_zero_jump(self) -> bool:
        return self.h_air <= 1e-9
