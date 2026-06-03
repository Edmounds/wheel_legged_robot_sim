"""Tests for QuinticTrajectory and JumpTrajectory."""
from __future__ import annotations

import numpy as np
import pytest

from sim.controllers.jump_trajectory import (
    GRAVITY,
    JumpTrajectory,
    JumpTrajectoryParams,
    QuinticTrajectory,
)


def test_quintic_respects_endpoint_position_velocity_acceleration() -> None:
    traj = QuinticTrajectory(
        h0=0.10, hd0=0.0, hdd0=0.0,
        hf=0.15, hdf=1.0, hddf=0.0,
        duration=0.1,
    )
    # 起点
    assert traj.height(0.0) == pytest.approx(0.10, abs=1e-9)
    assert traj.velocity(0.0) == pytest.approx(0.0, abs=1e-9)
    assert traj.acceleration(0.0) == pytest.approx(0.0, abs=1e-9)
    # 终点
    assert traj.height(0.1) == pytest.approx(0.15, abs=1e-6)
    assert traj.velocity(0.1) == pytest.approx(1.0, abs=1e-6)
    assert traj.acceleration(0.1) == pytest.approx(0.0, abs=1e-6)


def test_quintic_clips_out_of_range_time() -> None:
    traj = QuinticTrajectory(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, duration=0.2)
    # 超出 duration 取终点
    assert traj.height(0.5) == pytest.approx(traj.height(0.2), abs=1e-9)
    # 负时间取起点
    assert traj.height(-0.1) == pytest.approx(traj.height(0.0), abs=1e-9)


def test_quintic_zero_duration_rejected() -> None:
    with pytest.raises(ValueError):
        QuinticTrajectory(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, duration=0.0)


def test_jump_trajectory_adaptive_crouch_from_high_height() -> None:
    params = JumpTrajectoryParams()
    traj = JumpTrajectory(params, h_start=0.142, cmd_jump_amplitude=1.0)
    # 从 0.142 蹲 5cm 到 0.092
    assert traj.h_low == pytest.approx(0.092, abs=1e-6)
    # h_high = h_low + extend_stroke (0.092+0.045=0.137, 未撞 h_safe_high 上限)
    assert traj.h_high == pytest.approx(0.092 + params.extend_stroke, abs=1e-6)
    assert traj.h_high <= params.h_safe_high + 1e-9


def test_jump_trajectory_fixed_stroke_shifts_window_at_ceiling() -> None:
    params = JumpTrajectoryParams()
    # 高站姿 0.153: adaptive h_low=0.103, h_low+stroke=0.148 超过 h_safe_high=0.140,
    # 应封顶 h_high=0.140 并把窗口下移到 h_low=0.095, 保持 stroke=extend_stroke。
    traj = JumpTrajectory(params, h_start=0.153, cmd_jump_amplitude=1.0)
    assert traj.h_high == pytest.approx(params.h_safe_high, abs=1e-9)
    assert traj.h_high - traj.h_low == pytest.approx(params.extend_stroke, abs=1e-6)
    assert traj.h_low >= params.h_min - 1e-9


def test_jump_trajectory_fixed_stroke_constant_across_heights() -> None:
    params = JumpTrajectoryParams()
    # 固定行程的核心保证: 只要不撞 h_min/h_safe_high 边界, 各 cmd_height 起跳行程一致。
    for h_start in (0.100, 0.110, 0.120, 0.130):
        traj = JumpTrajectory(params, h_start=h_start, cmd_jump_amplitude=1.0)
        assert traj.h_high - traj.h_low == pytest.approx(params.extend_stroke, abs=1e-6)


def test_jump_trajectory_adaptive_crouch_clamped_to_h_min() -> None:
    params = JumpTrajectoryParams()
    traj = JumpTrajectory(params, h_start=0.080, cmd_jump_amplitude=1.0)
    # 0.080 - 0.05 < h_min=0.0785, 应该 clamp 到 h_min
    assert traj.h_low == pytest.approx(params.h_min, abs=1e-9)


def test_jump_trajectory_takeoff_velocity_from_air_height() -> None:
    params = JumpTrajectoryParams(air_height_max=0.10)
    # cmd_jump=1 → h_air = 0.10m → v_takeoff = sqrt(2 * 9.81 * 0.10)
    traj = JumpTrajectory(params, h_start=0.142, cmd_jump_amplitude=1.0)
    expected_v = float(np.sqrt(2.0 * GRAVITY * 0.10))
    assert traj.v_takeoff == pytest.approx(expected_v, abs=1e-6)

    # cmd_jump=0.5 → h_air = 0.05m → v_takeoff = sqrt(2 * 9.81 * 0.05)
    traj_half = JumpTrajectory(params, h_start=0.142, cmd_jump_amplitude=0.5)
    expected_v_half = float(np.sqrt(2.0 * GRAVITY * 0.05))
    assert traj_half.v_takeoff == pytest.approx(expected_v_half, abs=1e-6)


def test_jump_trajectory_extend_ends_with_takeoff_velocity() -> None:
    params = JumpTrajectoryParams()
    traj = JumpTrajectory(params, h_start=0.142, cmd_jump_amplitude=1.0)
    # extend 轨迹末速 = v_takeoff
    assert traj.extend.velocity(traj.extend.duration) == pytest.approx(traj.v_takeoff, abs=1e-6)
    # extend 起点静止
    assert traj.extend.velocity(0.0) == pytest.approx(0.0, abs=1e-9)


def test_jump_trajectory_zero_amplitude_means_no_jump() -> None:
    params = JumpTrajectoryParams()
    traj = JumpTrajectory(params, h_start=0.142, cmd_jump_amplitude=0.0)
    assert traj.is_zero_jump()
    assert traj.v_takeoff == 0.0


def test_jump_trajectory_setup_land_after_flight() -> None:
    params = JumpTrajectoryParams()
    traj = JumpTrajectory(params, h_start=0.142, cmd_jump_amplitude=1.0)
    assert traj.land is None
    traj.setup_land(h_contact=0.140, v_contact=-1.0)
    assert traj.land is not None
    assert traj.land.height(0.0) == pytest.approx(0.140, abs=1e-9)
    assert traj.land.velocity(0.0) == pytest.approx(-1.0, abs=1e-9)
    assert traj.land.height(traj.land.duration) == pytest.approx(params.h_stand_after_land, abs=1e-6)
    assert traj.land.velocity(traj.land.duration) == pytest.approx(0.0, abs=1e-6)
