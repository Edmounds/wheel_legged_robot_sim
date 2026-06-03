from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim.controllers.default_params import STAND_PARAMS, params_from_dict, params_to_dict


def test_combined_params_roundtrip_preserves_control_fields() -> None:
    params = replace(
        STAND_PARAMS,
        vmc=replace(STAND_PARAMS.vmc),
        q_diag=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        r_diag=np.array([7.0, 8.0]),
        target_velocity=0.31,
        pitch_lean_gain=0.23,
        velocity_ki=0.45,
        position_kp=1.7,
        position_kd=2.1,
        position_velocity_limit=0.4,
        yaw_damping=0.56,
        yaw_ki=0.67,
        target_yaw_rate=0.78,
        fixed_height=True,
        lqr_height_bin_size=0.025,
        ff_gain=4.5,
    )
    params.vmc.kp_land = 17.5
    params.vmc.kd_land = 4.2
    params.vmc.flight_pitch_kd = 1.7
    params.vmc.roll_level_kp_height = 0.031
    params.vmc.roll_level_kd_height = 0.006
    params.vmc.roll_level_offset_limit = 0.005
    params.vmc.slope_squat_margin = 0.009

    restored = params_from_dict(params_to_dict(params))

    np.testing.assert_allclose(restored.q_diag, params.q_diag)
    np.testing.assert_allclose(restored.r_diag, params.r_diag)
    assert restored.target_velocity == params.target_velocity
    assert restored.pitch_lean_gain == params.pitch_lean_gain
    assert restored.velocity_ki == params.velocity_ki
    assert restored.position_kp == params.position_kp
    assert restored.position_kd == params.position_kd
    assert restored.position_velocity_limit == params.position_velocity_limit
    assert restored.yaw_damping == params.yaw_damping
    assert restored.yaw_ki == params.yaw_ki
    assert restored.target_yaw_rate == params.target_yaw_rate
    assert restored.fixed_height is params.fixed_height
    assert restored.lqr_height_bin_size == params.lqr_height_bin_size
    assert restored.ff_gain == params.ff_gain
    assert restored.vmc.kp_land == params.vmc.kp_land
    assert restored.vmc.kd_land == params.vmc.kd_land
    assert restored.vmc.flight_pitch_kd == params.vmc.flight_pitch_kd
    assert restored.vmc.roll_level_kp_height == params.vmc.roll_level_kp_height
    assert restored.vmc.roll_level_kd_height == params.vmc.roll_level_kd_height
    assert restored.vmc.roll_level_offset_limit == params.vmc.roll_level_offset_limit
    assert restored.vmc.slope_squat_margin == params.vmc.slope_squat_margin


def test_params_from_dict_disables_roll_leveling_for_old_configs() -> None:
    data = params_to_dict(STAND_PARAMS)
    data["vmc"].pop("roll_level_kp_height")
    data["vmc"].pop("roll_level_kd_height")
    data["vmc"].pop("roll_level_offset_limit")

    restored = params_from_dict(data)

    assert restored.vmc.roll_level_kp_height == 0.0
    assert restored.vmc.roll_level_kd_height == 0.0
    assert restored.vmc.roll_level_offset_limit == 0.0


def test_params_from_dict_supplies_defaults_for_new_land_fields() -> None:
    """旧 JSON 不含 kp_land/kd_land/flight_pitch_kd/slope_squat_margin 时, fallback 到当前默认."""
    data = params_to_dict(STAND_PARAMS)
    data["vmc"].pop("kp_land")
    data["vmc"].pop("kd_land")
    data["vmc"].pop("flight_pitch_kd")
    data["vmc"].pop("slope_squat_margin")

    restored = params_from_dict(data)

    assert restored.vmc.kp_land == 15.0
    assert restored.vmc.kd_land == 3.5
    assert restored.vmc.flight_pitch_kd == 1.5
    assert restored.vmc.slope_squat_margin == 0.0


def test_stand_params_enable_roll_leveling_by_default() -> None:
    assert STAND_PARAMS.r_diag.tolist() == [200.0, 400.0]
    assert STAND_PARAMS.vmc.roll_level_kp_height == 0.0
    assert STAND_PARAMS.vmc.roll_level_kd_height == 0.002
    assert STAND_PARAMS.vmc.roll_level_offset_limit == 0.035
    assert STAND_PARAMS.vmc.slope_squat_margin == 0.005
