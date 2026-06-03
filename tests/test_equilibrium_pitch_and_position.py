"""Equilibrium pitch and position outer loop integration tests."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from sim.controllers.balance_lqr import equilibrium_pitch_from_geometry
from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
from sim.controllers.phase import JumpPhase, JumpPhaseMachine
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import extract_sim_state


def _stand_model(tmp_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model_path = prepare_controlled_mujoco_xml(Path("sim/robot/robot.urdf"), cache_dir=tmp_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert stand_id >= 0
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)
    return model, data


def test_equilibrium_pitch_matches_geometry(tmp_path: Path) -> None:
    """stand keyframe 下,几何稳态 pitch 应在 -0.148 rad 附近。

    2026-05-19: base_link inertial Y 由 0.00435 调到 0.000247 (~-4mm),
    让 LUT 配置 h=0.142 处 pitch_eq≈0。stand keyframe 的电机角 0.752 在
    LUT 范围 (theta_max=0.65) 之外,腿姿不同,所以这里残留 -0.148 rad。
    """
    model, data = _stand_model(tmp_path)
    pitch_eq = equilibrium_pitch_from_geometry(model, data)
    assert pitch_eq == pytest.approx(-0.148, abs=0.005)


def test_equilibrium_pitch_invariant_under_base_rotation(tmp_path: Path) -> None:
    """函数应只依赖关节构型(本体系几何),不依赖 base 在世界系的姿态。"""
    model, data = _stand_model(tmp_path)
    base = equilibrium_pitch_from_geometry(model, data)

    # 用归一化四元数把 base 整体绕世界 X 轴转 0.05 rad
    half = 0.025
    data.qpos[3:7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    mujoco.mj_forward(model, data)
    tilted = equilibrium_pitch_from_geometry(model, data)
    assert tilted == pytest.approx(base, abs=1e-4)


def test_position_anchor_set_on_stand_entry(tmp_path: Path) -> None:
    """从 LAND 进入 STAND 时,controller 锁住当前位置并清掉速度积分。"""
    model, data = _stand_model(tmp_path)
    phase_machine = JumpPhaseMachine()
    phase_machine.phase = JumpPhase.LAND
    controller = CombinedController(STAND_PARAMS, phase_machine)
    controller._velocity_integral = 1.23
    # 先走一步 LAND,初始化 LQR
    state = extract_sim_state(model, data)
    phase_machine.phase = JumpPhase.LAND
    controller._last_phase = JumpPhase.LAND
    # 切到 STAND 触发入口逻辑
    phase_machine.phase = JumpPhase.STAND
    controller(model, data, state)
    assert controller._position_anchor is not None
    np.testing.assert_allclose(controller._position_anchor, state.base_position[:2])
    assert controller._velocity_integral == 0.0


def test_position_anchor_released_when_driving(tmp_path: Path) -> None:
    """target_velocity != 0 时 anchor 应该被清掉,避免行驶模式被位置环拉回去。"""
    model, data = _stand_model(tmp_path)
    phase_machine = JumpPhaseMachine()
    controller = CombinedController(replace(STAND_PARAMS, target_velocity=0.5), phase_machine)
    state = extract_sim_state(model, data)
    controller(model, data, state)
    assert controller._position_anchor is None


def test_position_outer_loop_zero_at_anchor(tmp_path: Path) -> None:
    """位置在 anchor 处时,外环输出 0。"""
    model, data = _stand_model(tmp_path)
    phase_machine = JumpPhaseMachine()
    controller = CombinedController(STAND_PARAMS, phase_machine)
    state = extract_sim_state(model, data)
    controller._position_anchor = np.array(state.base_position[:2], dtype=float)
    vel_correction = controller._position_outer_loop(model, data, state)
    assert vel_correction == pytest.approx(0.0, abs=1e-9)


def test_position_outer_loop_clamps_to_limit(tmp_path: Path) -> None:
    """位置误差远大时,外环输出被 position_velocity_limit 限幅。"""
    model, data = _stand_model(tmp_path)
    phase_machine = JumpPhaseMachine()
    controller = CombinedController(STAND_PARAMS, phase_machine)
    state = extract_sim_state(model, data)
    # 模拟机器人远离 anchor 5 米 (沿 +Y 方向),应该被位置环拉回 → 负 vel_correction
    far_anchor = state.base_position[:2].copy()
    far_anchor[1] -= 5.0
    controller._position_anchor = far_anchor
    vel_correction = controller._position_outer_loop(model, data, state)
    assert vel_correction == pytest.approx(-STAND_PARAMS.position_velocity_limit, abs=1e-6)
