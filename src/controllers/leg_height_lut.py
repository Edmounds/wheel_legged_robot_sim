"""Lookup table for ``cmd_height → target motor angle`` inverse kinematics.

Generated offline by ``scripts/probe_leg_geometry.py``. Runtime VMC uses
``motor_angle_from_height`` to translate user height commands into joint-space
setpoints, replacing task-space height feedback which oscillated near the
four-bar linkage's kinematic singularity.

Reload by re-running the probe whenever the URDF, mjcf_builder processing, or
mesh preprocessing changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_LUT_PATH = Path(__file__).resolve().parent / "leg_height_lut.json"


@dataclass(frozen=True)
class LegHeightLUT:
    theta_grid: np.ndarray  # average of left/right active motor angles
    theta_left_grid: np.ndarray
    theta_right_grid: np.ndarray
    h_grid: np.ndarray  # paired with theta grids; must be monotonic
    dy_grid: np.ndarray  # wheel midpoint local-Y in base frame, paired with h_grid
    pitch_eq_grid: np.ndarray  # CoM-derived equilibrium pitch, paired with h_grid
    h_min: float
    h_max: float
    theta_min: float
    theta_max: float

    @classmethod
    def from_json(cls, path: Path = _LUT_PATH) -> "LegHeightLUT":
        with open(path) as f:
            data = json.load(f)
        theta_grid = np.asarray(data["theta_grid"], dtype=float)
        theta_left_grid = np.asarray(data.get("theta_left_grid", theta_grid), dtype=float)
        theta_right_grid = np.asarray(data.get("theta_right_grid", theta_grid), dtype=float)
        h_grid = np.asarray(data["h_grid"], dtype=float)
        dy_grid = np.asarray(data.get("dy_grid", np.zeros_like(theta_grid)), dtype=float)
        pitch_eq_grid = np.asarray(data.get("pitch_eq_grid", np.zeros_like(theta_grid)), dtype=float)
        if theta_grid.shape != h_grid.shape or theta_grid.ndim != 1:
            raise ValueError("LUT theta_grid and h_grid must be 1-D arrays of equal length")
        if theta_left_grid.shape != theta_grid.shape or theta_right_grid.shape != theta_grid.shape:
            raise ValueError("LUT side theta grids must match theta_grid")
        if dy_grid.shape != theta_grid.shape or pitch_eq_grid.shape != theta_grid.shape:
            raise ValueError("LUT dy_grid and pitch_eq_grid must match theta_grid shape")
        h_diffs = np.diff(h_grid)
        if not (np.all(h_diffs > 0) or np.all(h_diffs < 0)):
            raise ValueError("LUT h_grid must be strictly monotonic")
        return cls(
            theta_grid=theta_grid,
            theta_left_grid=theta_left_grid,
            theta_right_grid=theta_right_grid,
            h_grid=h_grid,
            dy_grid=dy_grid,
            pitch_eq_grid=pitch_eq_grid,
            h_min=float(data["h_min"]),
            h_max=float(data["h_max"]),
            theta_min=float(data["theta_min"]),
            theta_max=float(data["theta_max"]),
        )

    def motor_angle_from_height(self, h: float, side: str | None = None) -> float:
        """Return motor angle that produces ``h`` (or the closest in-range h)."""
        return self._interp_by_height(h, self._theta_grid_for_side(side))

    def motor_angle_dh(self, h: float, side: str | None = None) -> float:
        """Return dθ/dh at ``h`` from the LUT curve."""
        gradients = np.gradient(self._theta_grid_for_side(side), self.h_grid)
        return self._interp_by_height(h, gradients)

    def height_dtheta(self, h: float, side: str | None = None) -> float:
        """Return dh/dθ at ``h`` from the LUT curve."""
        gradients = np.gradient(self.h_grid, self._theta_grid_for_side(side))
        return self._interp_by_height(h, gradients)

    def dy_wheel_dh(self, h: float) -> float:
        """Return d(wheel_y_in_base)/dh at ``h`` from the LUT curve."""
        degree = min(3, len(self.h_grid) - 1)
        coefficients = np.polyfit(self.h_grid, self.dy_grid, degree)
        derivative = np.polyder(coefficients)
        h_clamped = float(np.clip(h, self.h_min, self.h_max))
        return float(np.polyval(derivative, h_clamped))

    def pitch_eq_from_height(self, h: float) -> float:
        """Return the CoM-derived equilibrium pitch for leg height ``h``."""
        return self._interp_by_height(h, self.pitch_eq_grid)

    def _theta_grid_for_side(self, side: str | None) -> np.ndarray:
        if side is None:
            return self.theta_grid
        if side == "left":
            return self.theta_left_grid
        if side == "right":
            return self.theta_right_grid
        raise ValueError(f"unknown leg side: {side}")

    def _interp_by_height(self, h: float, values: np.ndarray) -> float:
        h_clamped = float(np.clip(h, self.h_min, self.h_max))
        h_increasing = self.h_grid[-1] > self.h_grid[0]
        if h_increasing:
            return float(np.interp(h_clamped, self.h_grid, values))
        return float(np.interp(h_clamped, self.h_grid[::-1], values[::-1]))


DEFAULT_LUT = LegHeightLUT.from_json()
