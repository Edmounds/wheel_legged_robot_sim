from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.controllers.jump_trajectory import JumpTrajectory


class JumpPhase(str, Enum):
    STAND = "stand"
    CROUCH = "crouch"
    EXTEND = "extend"
    FLIGHT = "flight"
    LAND = "land"
    FALLEN = "fallen"


@dataclass(frozen=True)
class JumpPhaseParams:
    """相位时长由 JumpTrajectory 提供; 这里只放 fallback / hard guard 参数。"""
    flight_timeout: float = 0.6
    pitch_fallen_threshold: float = 1.0
    # 起跳确认需要 ncon=0 持续的最小时间 (s)。短于此视为弹跳,继续 EXTEND 推。
    takeoff_airborne_confirm: float = 0.005


class JumpPhaseMachine:
    def __init__(self, params: JumpPhaseParams | None = None) -> None:
        self.params = params or JumpPhaseParams()
        self.phase = JumpPhase.STAND
        self.time_in_phase = 0.0
        self.trajectory: JumpTrajectory | None = None
        self._airborne_time = 0.0  # EXTEND 期间累计的 ncon=0 时长

    def start_jump(self, trajectory: JumpTrajectory) -> None:
        """触发跳跃 (必须传入预规划好的 trajectory)。"""
        if self.phase == JumpPhase.STAND:
            self.trajectory = trajectory
            self._set_phase(JumpPhase.CROUCH)

    def update(
        self,
        *,
        dt: float,
        contact_count: int,
        leg_height: float | None = None,
        vz: float | None = None,
        pitch: float | None = None,
    ) -> JumpPhase:
        del leg_height  # 保留参数兼容性,当前不使用
        if self.phase == JumpPhase.FALLEN:
            return self.phase

        self.time_in_phase += dt

        if pitch is not None and abs(float(pitch)) > self.params.pitch_fallen_threshold:
            self._set_phase(JumpPhase.FALLEN)
            return self.phase

        if self.trajectory is None:
            # 没有 trajectory 应该只发生在 STAND/FALLEN
            return self.phase

        # 累计 EXTEND 期间的"持续离地"时长
        if self.phase == JumpPhase.EXTEND:
            if contact_count == 0:
                self._airborne_time += dt
            else:
                self._airborne_time = 0.0

        if self.phase == JumpPhase.CROUCH and self.time_in_phase >= self.trajectory.crouch.duration:
            self._set_phase(JumpPhase.EXTEND)
        elif self.phase == JumpPhase.EXTEND:
            # 起跳确认: ncon=0 持续 >= takeoff_airborne_confirm (5ms 默认),
            # AND vz > 0.3 (确实在上升)。短于此视为弹跳,继续 EXTEND 推。
            sustained_airborne = self._airborne_time >= self.params.takeoff_airborne_confirm
            rising = vz is not None and vz > 0.3
            if sustained_airborne and rising:
                self._set_phase(JumpPhase.FLIGHT)
            elif self.time_in_phase >= self.trajectory.extend.duration * 2.0:
                # duration 跑了 2 倍仍没起跳 → 强制 LAND (短跳 / 能量不够)。
                # 2x 而非 1x 让 motor 有时间在弹跳间隙补推。
                self._set_phase(JumpPhase.LAND)
        elif self.phase == JumpPhase.FLIGHT:
            if contact_count > 0 and (vz is None or vz <= 0.0):
                self._set_phase(JumpPhase.LAND)
                # 入 LAND 瞬间需要外部 setup_land (vmc 调用)
            elif self.time_in_phase >= self.params.flight_timeout:
                self._set_phase(JumpPhase.LAND)
        elif (
            self.phase == JumpPhase.LAND
            and self.trajectory.land is not None
            and self.time_in_phase >= self.trajectory.land.duration
            and contact_count > 0
            and (vz is None or abs(vz) < 0.2)
        ):
            self.trajectory = None
            self._set_phase(JumpPhase.STAND)
        return self.phase

    def _set_phase(self, phase: JumpPhase) -> None:
        if self.phase != phase:
            self.phase = phase
            self.time_in_phase = 0.0
            self._airborne_time = 0.0
