from __future__ import annotations

import json

import numpy as np

from sim.controllers.leg_height_lut import DEFAULT_LUT, LegHeightLUT


def test_default_lut_contains_geometry_feedforward_fields() -> None:
    assert DEFAULT_LUT.dy_grid.shape == DEFAULT_LUT.theta_grid.shape
    assert DEFAULT_LUT.pitch_eq_grid.shape == DEFAULT_LUT.theta_grid.shape
    assert DEFAULT_LUT.theta_left_grid.shape == DEFAULT_LUT.theta_grid.shape
    assert DEFAULT_LUT.theta_right_grid.shape == DEFAULT_LUT.theta_grid.shape
    assert np.all(np.isfinite(DEFAULT_LUT.dy_grid))
    assert np.all(np.isfinite(DEFAULT_LUT.pitch_eq_grid))


def test_default_lut_exposes_viewer_low_range() -> None:
    assert np.isclose(DEFAULT_LUT.h_min, 0.07843, atol=1e-4)
    assert DEFAULT_LUT.h_max > 0.142
    assert np.isclose(DEFAULT_LUT.motor_angle_from_height(DEFAULT_LUT.h_min, side="left"), -0.1099, atol=1e-4)
    assert np.isclose(DEFAULT_LUT.motor_angle_from_height(DEFAULT_LUT.h_min, side="right"), -0.1100, atol=1e-4)
    assert DEFAULT_LUT.motor_angle_from_height(0.142) < 0.65


def test_default_lut_geometry_values_match_side_specific_probe() -> None:
    assert np.isclose(DEFAULT_LUT.pitch_eq_from_height(0.142), 0.0003, atol=0.005)
    assert np.isclose(DEFAULT_LUT.dy_wheel_dh(0.142), 0.243, atol=0.04)


def test_lut_uses_side_specific_motor_angles(tmp_path) -> None:
    path = tmp_path / "lut.json"
    path.write_text(
        json.dumps(
            {
                "theta_grid": [0.0, 1.0, 2.0],
                "theta_left_grid": [0.0, 1.0, 2.0],
                "theta_right_grid": [0.2, 1.2, 2.2],
                "h_grid": [0.10, 0.20, 0.30],
                "dy_grid": [0.0, 0.5, 1.0],
                "pitch_eq_grid": [-0.1, -0.2, -0.3],
                "h_min": 0.10,
                "h_max": 0.30,
                "theta_min": 0.0,
                "theta_max": 2.0,
            }
        )
    )

    lut = LegHeightLUT.from_json(path)

    assert np.isclose(lut.motor_angle_from_height(0.15, side="left"), 0.5)
    assert np.isclose(lut.motor_angle_from_height(0.15, side="right"), 0.7)
    with np.testing.assert_raises_regex(ValueError, "unknown leg side"):
        lut.motor_angle_from_height(0.15, side="front")


def test_lut_interpolates_height_derivatives_from_json(tmp_path) -> None:
    path = tmp_path / "lut.json"
    path.write_text(
        json.dumps(
            {
                "theta_grid": [0.0, 1.0, 2.0],
                "h_grid": [0.10, 0.20, 0.30],
                "dy_grid": [0.0, 0.5, 1.0],
                "pitch_eq_grid": [-0.1, -0.2, -0.3],
                "h_min": 0.10,
                "h_max": 0.30,
                "theta_min": 0.0,
                "theta_max": 2.0,
            }
        )
    )

    lut = LegHeightLUT.from_json(path)

    assert np.isclose(lut.motor_angle_from_height(0.15), 0.5)
    assert np.isclose(lut.height_dtheta(0.15), 0.1)
    assert np.isclose(lut.motor_angle_dh(0.15), 10.0)
    assert np.isclose(lut.dy_wheel_dh(0.15), 5.0)
    assert np.isclose(lut.pitch_eq_from_height(0.15), -0.15)
