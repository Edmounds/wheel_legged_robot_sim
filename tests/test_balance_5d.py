from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from sim.controllers.balance_lqr import (
    _reduced_balance_system_5d,
    compute_balance_lqr_gain_5d,
)
from sim.controllers.balance_state import (
    balance_tangent_state_5d,
    balance_tangent_state_6d,
)
from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
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


def test_balance_tangent_state_5d_drops_wheel_pos_only(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)
    state = extract_sim_state(model, data)

    five = balance_tangent_state_5d(model, data, state)
    six = balance_tangent_state_6d(model, data, state)

    assert five.shape == (5,)
    # 5D = 6D 删掉 index 4 (wheel_pos): [pitch, pitch_rate, roll, roll_rate, wheel_vel]
    np.testing.assert_allclose(five, np.array([six[0], six[1], six[2], six[3], six[5]]))


def test_reduced_balance_system_5d_shape_and_stabilizable(tmp_path: Path) -> None:
    model, data = _stand_model(tmp_path)

    a, b = _reduced_balance_system_5d(model, data)
    q_diag = np.array([1000.0, 200.0, 1000.0, 200.0, 500.0])
    r_diag = np.array([200.0, 400.0])
    gain = compute_balance_lqr_gain_5d(model, data, q_diag, r_diag)
    eigenvalues = np.linalg.eigvals(a - b @ gain)

    assert a.shape == (5, 5)
    assert b.shape == (5, 2)
    assert gain.shape == (2, 5)
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(b))
    assert np.all(np.isfinite(gain))
    # closed-loop eigenvalues 全部在单位圆内 = 离散时间稳定
    assert np.max(np.abs(eigenvalues)) < 1.0


def test_reduced_balance_system_5d_matches_6d_minus_wheel_pos(tmp_path: Path) -> None:
    from sim.controllers.balance_lqr import _reduced_balance_system

    model, data = _stand_model(tmp_path)
    a6, b6 = _reduced_balance_system(model, data)
    a5, b5 = _reduced_balance_system_5d(model, data)

    # 5D = 6D 删掉 row/col 4 (wheel_pos)。验证保留的元素一致：
    # 6D index [0,1,2,3,5] → 5D index [0,1,2,3,4]
    keep = [0, 1, 2, 3, 5]
    np.testing.assert_allclose(a5, a6[np.ix_(keep, keep)])
    np.testing.assert_allclose(b5, b6[keep, :])


def test_compute_balance_lqr_gain_5d_rejects_wrong_q_shape(tmp_path: Path) -> None:
    import pytest

    model, data = _stand_model(tmp_path)
    bad_q = np.array([1000.0, 200.0, 1000.0, 200.0, 50.0, 500.0])
    r_diag = np.array([200.0, 400.0])

    with pytest.raises(ValueError, match="q_diag"):
        compute_balance_lqr_gain_5d(model, data, bad_q, r_diag)




def test_combined_controller_prewarms_height_bins_without_online_lqr_solves(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import sim.controllers.balance_lqr as balance_lqr

    model, data = _stand_model(tmp_path)
    vmc_params = replace(
        STAND_PARAMS.vmc,
        roll_level_kp_height=0.0,
        roll_level_kd_height=0.0,
        roll_level_offset_limit=0.0,
    )
    params = replace(
        STAND_PARAMS,
        vmc=vmc_params,
        q_diag=STAND_PARAMS.q_diag.copy(),
        r_diag=STAND_PARAMS.r_diag.copy(),
        fixed_height=False,
    )
    controller = CombinedController(params)
    original = balance_lqr.compute_balance_lqr_gain_5d
    calls = 0

    def counted_gain(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(balance_lqr, "compute_balance_lqr_gain_5d", counted_gain)

    state = extract_sim_state(model, data)
    controller(model, data, state)
    first_call_count = calls
    assert first_call_count == 1
    active_roll_bin = controller._roll_bin(state)

    test_heights = np.linspace(params.vmc.lut.h_min, params.vmc.lut.h_max, 8)
    for height in test_heights:
        state = replace(
            state,
            base_position=np.array([
                state.base_position[0],
                state.base_position[1],
                height,
            ]),
        )
        assert controller._roll_bin(state) == active_roll_bin
        controller._interpolated_lqr_inputs(model, data, state)

    assert calls == first_call_count
