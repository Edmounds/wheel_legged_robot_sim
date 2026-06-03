from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from src.state import SimState


TangentStateProvider = Callable[[mujoco.MjModel, mujoco.MjData, SimState], np.ndarray]


@dataclass
class LqrController:
    gain: np.ndarray
    target: np.ndarray
    feedforward: np.ndarray
    tangent_state_provider: TangentStateProvider | None = None

    @classmethod
    def zero_gain(cls, model: mujoco.MjModel) -> LqrController:
        return cls(
            gain=np.zeros((model.nu, 2 * model.nv)),
            target=np.zeros(2 * model.nv),
            feedforward=np.zeros(model.nu),
        )

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData, state: SimState) -> np.ndarray:
        if not np.all(np.isfinite(self.gain)) or not np.all(np.isfinite(self.target)) or not np.all(np.isfinite(self.feedforward)):
            raise ValueError("LQR gain, target, and feedforward must be finite")
        if self.tangent_state_provider is None:
            if np.any(self.gain):
                raise ValueError("tangent_state_provider is required for nonzero LQR gain")
            current = np.zeros(self.gain.shape[1])
        else:
            current = self.tangent_state_provider(model, data, state)
        if current.shape != self.target.shape:
            raise ValueError(f"LQR state shape must be {self.target.shape}, got {current.shape}")
        if not np.all(np.isfinite(current)):
            raise ValueError("LQR state must be finite")
        raw_control = self.feedforward - self.gain @ (current - self.target)
        if raw_control.shape == (model.nu,):
            clipped = np.clip(raw_control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        else:
            clipped = raw_control
        if not np.all(np.isfinite(clipped)):
            raise ValueError("LQR control must be finite")
        return clipped


def solve_discrete_lqr(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    p = solve_discrete_are(a, b, q, r)
    return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
