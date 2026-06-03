"""Heading-hold (航向保持) outer-loop tests.

航向保持与位置外环 (test_equilibrium_pitch_and_position.py) 同构: 未发转向指令
(target_yaw_rate≈0) 且站立接地时, 锁定当前航向并用 P 外环抵抗外部扰动回正。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim.controllers.combined import CombinedController, _wrap_to_pi
from sim.controllers.default_params import (
    STAND_PARAMS,
    STAND_THEN_DRIVE_PARAMS,
    params_from_dict,
    params_to_dict,
)
from sim.controllers.phase import JumpPhase, JumpPhaseMachine
from sim.launch_mujoco import step_controlled_model
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


def _rotate_base_about_world_z(model: mujoco.MjModel, data: mujoco.MjData, angle: float) -> None:
    """绕世界 Z 轴预乘一个旋转 (extrinsic), 纯航向扰动, 不改变 pitch/roll 与重力对齐。"""
    addresses = model_addresses(model)
    base = addresses.root_qpos + 3
    q = np.array(data.qpos[base : base + 4], dtype=float)
    dq = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)], dtype=float)
    res = np.zeros(4)
    mujoco.mju_mulQuat(res, dq, q)
    data.qpos[base : base + 4] = res
    mujoco.mj_forward(model, data)


# ---------- 参数预设 / 序列化 ----------

def test_stand_params_enable_heading_hold_by_default() -> None:
    assert STAND_PARAMS.heading_hold_kp == 2.0
    assert STAND_PARAMS.heading_hold_rate_limit == 0.8
    assert STAND_THEN_DRIVE_PARAMS.heading_hold_kp == 2.0
    assert STAND_THEN_DRIVE_PARAMS.heading_hold_rate_limit == 0.8


def test_heading_hold_serialization_roundtrip() -> None:
    params = replace(STAND_PARAMS, heading_hold_kp=1.7, heading_hold_rate_limit=0.55)
    restored = params_from_dict(params_to_dict(params))
    assert restored.heading_hold_kp == 1.7
    assert restored.heading_hold_rate_limit == 0.55


def test_heading_hold_defaults_off_for_old_configs() -> None:
    """旧 JSON 不含 heading_hold_* 时, fallback 到关闭 (kp=0), 保持旧 yaw-rate 阻尼行为。"""
    data = params_to_dict(STAND_PARAMS)
    data.pop("heading_hold_kp")
    data.pop("heading_hold_rate_limit")
    restored = params_from_dict(data)
    assert restored.heading_hold_kp == 0.0
    assert restored.heading_hold_rate_limit == 1.0


# ---------- anchor 生命周期 ----------

def test_heading_anchor_set_on_stand_entry(tmp_path: Path) -> None:
    """从 LAND 进入 STAND 时锁定当前航向。"""
    model, data = _stand_model(tmp_path)
    phase_machine = JumpPhaseMachine()
    controller = CombinedController(STAND_PARAMS, phase_machine)
    controller._last_phase = JumpPhase.LAND
    phase_machine.phase = JumpPhase.STAND
    state = extract_sim_state(model, data)
    controller(model, data, state)
    assert controller._heading_anchor is not None
    assert controller._heading_anchor == pytest.approx(controller._base_heading(model, data))


def test_heading_anchor_released_when_turning(tmp_path: Path) -> None:
    """target_yaw_rate != 0 时丢弃 anchor, 让转向指令不被航向环拉回。"""
    model, data = _stand_model(tmp_path)
    controller = CombinedController(replace(STAND_PARAMS, target_yaw_rate=0.5))
    state = extract_sim_state(model, data)
    controller(model, data, state)
    assert controller._heading_anchor is None


def test_heading_anchor_recaptured_when_yaw_released(tmp_path: Path) -> None:
    """松开转向 (yaw≈0) 后重新锁定当前航向 (anchor 之前为 None)。"""
    model, data = _stand_model(tmp_path)
    controller = CombinedController(STAND_PARAMS)
    controller._heading_anchor = None
    state = extract_sim_state(model, data)
    controller(model, data, state)
    assert controller._heading_anchor is not None


# ---------- 外环数学: 零点 / 限幅 / 符号 / 开关 ----------

def test_heading_outer_loop_zero_at_anchor(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)
    controller = CombinedController(STAND_PARAMS)
    controller._heading_anchor = controller._base_heading(model, data)
    assert controller._heading_outer_loop(model, data) == pytest.approx(0.0, abs=1e-9)


def test_heading_outer_loop_sign_drives_back_to_anchor(tmp_path: Path) -> None:
    """anchor 在当前航向 +0.1 rad: 误差为正, 输出正的 yaw-rate 参考 (回正方向)。"""
    model, data = _stand_model(tmp_path)
    controller = CombinedController(STAND_PARAMS)
    controller._heading_anchor = controller._base_heading(model, data) + 0.1
    out = controller._heading_outer_loop(model, data)
    assert out == pytest.approx(STAND_PARAMS.heading_hold_kp * 0.1, abs=1e-9)
    assert out > 0.0


def test_heading_outer_loop_clamps_to_rate_limit(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)
    controller = CombinedController(STAND_PARAMS)
    controller._heading_anchor = controller._base_heading(model, data) + 1.0
    out = controller._heading_outer_loop(model, data)
    assert out == pytest.approx(STAND_PARAMS.heading_hold_rate_limit, abs=1e-6)


def test_heading_outer_loop_off_when_kp_zero(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)
    controller = CombinedController(replace(STAND_PARAMS, heading_hold_kp=0.0))
    controller._heading_anchor = controller._base_heading(model, data) + 0.1
    assert controller._heading_outer_loop(model, data) == 0.0


def test_heading_outer_loop_off_when_turning(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)
    controller = CombinedController(replace(STAND_PARAMS, target_yaw_rate=0.5))
    controller._heading_anchor = controller._base_heading(model, data) + 0.1
    assert controller._heading_outer_loop(model, data) == 0.0


# ---------- 串级: 航向参考进入 yaw 阻尼内环 ----------

def test_heading_ref_shifts_effective_target_yaw_rate(tmp_path: Path) -> None:
    """heading_rate_ref 抬高 effective target, 使 yaw_error 减少 yaw_damping*ref。"""
    model, data = _stand_model(tmp_path)
    params = replace(STAND_PARAMS, yaw_damping=1.0, yaw_ki=0.0, target_yaw_rate=0.0)
    controller = CombinedController(params)
    state = extract_sim_state(model, data)
    dt = float(model.opt.timestep)
    base = controller._compute_yaw_correction(state, dt, 0.0)
    with_ref = controller._compute_yaw_correction(state, dt, 0.5)
    assert (with_ref - base) == pytest.approx(-params.yaw_damping * 0.5, abs=1e-9)


# ---------- 闭环行为: 扰动后回到航向 ----------

def test_heading_hold_recovers_after_yaw_disturbance(tmp_path: Path) -> None:
    """注入纯航向扰动后, 开启航向保持的机器人显著回正; 关闭时几乎不回正。

    扰动是绕世界 Z 的静态旋转 (角速度≈0)。纯 yaw-rate 阻尼 (kp=0) 看到 yaw_rate≈0
    不产生回正力矩, 航向停在偏置处; 航向保持 (kp>0) 把航向误差转成 yaw-rate 参考拉回。
    """
    settle_steps = 300
    recover_steps = 900
    disturbance = 0.15

    def run(heading_hold_kp: float) -> tuple[float, float]:
        model, data = _stand_model(tmp_path)
        params = replace(STAND_PARAMS, heading_hold_kp=heading_hold_kp)
        controller = CombinedController(params, JumpPhaseMachine())
        for _ in range(settle_steps):
            assert step_controlled_model(model, data, controller)
        anchor = controller._heading_anchor
        assert anchor is not None
        _rotate_base_about_world_z(model, data, disturbance)
        err_kick = abs(_wrap_to_pi(anchor - controller._base_heading(model, data)))
        for _ in range(recover_steps):
            assert step_controlled_model(model, data, controller)
        err_final = abs(_wrap_to_pi(anchor - controller._base_heading(model, data)))
        return err_kick, err_final

    err_kick_hold, err_final_hold = run(STAND_PARAMS.heading_hold_kp)
    assert err_kick_hold == pytest.approx(disturbance, abs=0.03)
    # 开启航向保持: 至少回正一半。
    assert err_final_hold < 0.5 * err_kick_hold

    _, err_final_off = run(0.0)
    # 关闭航向保持: 回正显著弱于开启 (静态偏置基本保留)。
    assert err_final_hold < err_final_off
