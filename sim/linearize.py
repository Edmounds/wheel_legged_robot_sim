from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class LinearizationConfig:
    epsilon: float = 1e-6


@dataclass(frozen=True)
class LinearizationResult:
    a: np.ndarray
    b: np.ndarray
    valid: bool
    reason: str | None = None
    nominal_next_state: np.ndarray | None = None
    model: mujoco.MjModel | None = None

    def transition_error(self, next_state: np.ndarray) -> np.ndarray:
        if self.nominal_next_state is None or self.model is None:
            raise ValueError("linearization result does not include a nominal transition")
        return _state_difference(self.model, self.nominal_next_state, next_state)


def one_step_transition(model: mujoco.MjModel, state: np.ndarray, control: np.ndarray) -> np.ndarray:
    _validate_state_control(model, state, control)
    data = mujoco.MjData(model)
    data.qpos[:] = state[: model.nq]
    data.qvel[:] = state[model.nq : model.nq + model.nv]
    data.ctrl[:] = control
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    return np.concatenate([data.qpos.copy(), data.qvel.copy()])


def linearize_transition(
    model: mujoco.MjModel,
    state: np.ndarray,
    control: np.ndarray,
    config: LinearizationConfig | None = None,
) -> LinearizationResult:
    config = config or LinearizationConfig()
    nx = 2 * model.nv
    nu = model.nu
    if not np.isfinite(config.epsilon) or config.epsilon <= 0.0:
        return LinearizationResult(np.zeros((nx, nx)), np.zeros((nx, nu)), valid=False, reason="invalid epsilon")
    if state.shape != (model.nq + model.nv,) or control.shape != (nu,) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(control)):
        return LinearizationResult(np.zeros((nx, nx)), np.zeros((nx, nu)), valid=False, reason="nonfinite input")

    nominal_next = one_step_transition(model, state, control)
    a = np.zeros((nx, nx))
    b = np.zeros((nx, nu))

    for index in range(nx):
        plus_state = _apply_tangent_delta(model, state, index, config.epsilon)
        minus_state = _apply_tangent_delta(model, state, index, -config.epsilon)
        plus_error = _state_difference(model, nominal_next, one_step_transition(model, plus_state, control))
        minus_error = _state_difference(model, nominal_next, one_step_transition(model, minus_state, control))
        if not np.all(np.isfinite(plus_error)) or not np.all(np.isfinite(minus_error)):
            return LinearizationResult(a, b, valid=False, reason="nonfinite transition")
        a[:, index] = (plus_error - minus_error) / (2.0 * config.epsilon)

    for index in range(nu):
        delta = np.zeros(nu)
        delta[index] = config.epsilon
        plus_error = _state_difference(model, nominal_next, one_step_transition(model, state, control + delta))
        minus_error = _state_difference(model, nominal_next, one_step_transition(model, state, control - delta))
        if not np.all(np.isfinite(plus_error)) or not np.all(np.isfinite(minus_error)):
            return LinearizationResult(a, b, valid=False, reason="nonfinite transition")
        b[:, index] = (plus_error - minus_error) / (2.0 * config.epsilon)

    return LinearizationResult(a, b, valid=True, nominal_next_state=nominal_next, model=model)


def _apply_tangent_delta(model: mujoco.MjModel, state: np.ndarray, index: int, value: float) -> np.ndarray:
    qpos = state[: model.nq].copy()
    qvel = state[model.nq : model.nq + model.nv].copy()
    if index < model.nv:
        tangent = np.zeros(model.nv)
        tangent[index] = value
        mujoco.mj_integratePos(model, qpos, tangent, 1.0)
    else:
        qvel[index - model.nv] += value
    return np.concatenate([qpos, qvel])


def _state_difference(model: mujoco.MjModel, reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    qpos_error = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, qpos_error, 1.0, reference[: model.nq], actual[: model.nq])
    qvel_error = actual[model.nq : model.nq + model.nv] - reference[model.nq : model.nq + model.nv]
    return np.concatenate([qpos_error, qvel_error])


def _validate_state_control(model: mujoco.MjModel, state: np.ndarray, control: np.ndarray) -> None:
    if state.shape != (model.nq + model.nv,):
        raise ValueError(f"state shape must be {(model.nq + model.nv,)}, got {state.shape}")
    if control.shape != (model.nu,):
        raise ValueError(f"control shape must be {(model.nu,)}, got {control.shape}")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(control)):
        raise ValueError("state and control must be finite")
