#!/usr/bin/env python3
"""Generate sim/controllers/leg_height_lut.json from analytic four-bar geometry.

The dynamic probe drives the active leg motors through MuJoCo's soft equality
constraint and records the settled pose. Near the lower leg limit that captures
a dynamic floor, not the mechanism's geometric range. This generator solves the
planar four-bar closure directly and uses ``mj_forward`` only to read the same
derived fields already stored in the LUT.

The low branch is calibrated from a manually checked MuJoCo viewer posture.
The LUT stores left and right motor qpos separately because mirrored joints can
show different qpos values for the same physical leg position.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.controllers.vmc import LEG_CLOSED_LOOP
from sim.model_semantics import MODEL_SEMANTICS
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import body_id, equality_id, model_addresses


ANALYTIC_BRANCH_THETA_MIN = -0.110000
THETA_SCAN_MIN = ANALYTIC_BRANCH_THETA_MIN
THETA_MAX = 0.65
THETA_STEP = 0.005
STAND_KEYFRAME_THETA = 0.752
CONNECT_TOLERANCE = 1e-5
VIEWER_LOW_QPOS = {
    "base_link_旋转-1": -0.110,
    "link1_right_旋转-5": -0.171,
    "link2_right_旋转-12": 0.0612,
    "base_link_旋转-2": -0.110,
    "link1_left_旋转-6": -0.171,
    "link2_left_旋转-13": 0.0611,
    "base_link_旋转-3": -0.108,
    "base_link_旋转-4": -0.108,
}


@dataclass(frozen=True)
class LegClosure:
    side: str
    motor_joint_id: int
    passive_joint_id: int
    elbow_joint_id: int
    wheel_joint_id: int
    equality_body1_id: int
    equality_body2_id: int
    equality_anchor1: np.ndarray
    equality_anchor2: np.ndarray
    a_world: np.ndarray
    ex: np.ndarray
    ez: np.ndarray
    a: np.ndarray
    e: np.ndarray
    r1: float
    r2c: float
    r2w: float
    r3: float
    wbc_angle: float
    motor_offset: float
    passive_offset: float
    elbow_offset: float
    branch_sign: int


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"zero-length vector while computing {name}")
    return np.asarray(vector, dtype=float) / norm


def _angle_wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _nearest_equivalent_angle(angle: float, reference: float) -> float:
    return float(angle + round((reference - angle) / (2.0 * np.pi)) * 2.0 * np.pi)


def _wheel_joint_name(side: str) -> str:
    return MODEL_SEMANTICS.wheel_joints[0 if side == "left" else 1]


def _body_local_point_world(
    data: mujoco.MjData,
    body_id_value: int,
    local_point: np.ndarray,
) -> np.ndarray:
    return np.asarray(data.xpos[body_id_value], dtype=float) + np.asarray(
        data.xmat[body_id_value], dtype=float
    ).reshape(3, 3) @ local_point


def _base_frame_geometry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_id: int,
    wheel_ids: dict[str, int],
) -> tuple[float, float]:
    base_p = np.asarray(data.xpos[base_id], dtype=float)
    base_r = np.asarray(data.xmat[base_id], dtype=float).reshape(3, 3)
    wheel_mid_w = 0.5 * (
        np.asarray(data.xipos[wheel_ids["left"]], dtype=float)
        + np.asarray(data.xipos[wheel_ids["right"]], dtype=float)
    )
    masses = np.asarray(model.body_mass, dtype=float)
    com_w = (masses[:, None] * np.asarray(data.xipos, dtype=float)).sum(axis=0) / float(
        np.sum(masses)
    )

    wheel_in_base = base_r.T @ (wheel_mid_w - base_p)
    com_in_base = base_r.T @ (com_w - base_p)
    com_from_wheel = com_in_base - wheel_in_base
    pitch_eq = float(np.atan2(-com_from_wheel[1], abs(com_from_wheel[2])))
    return float(wheel_in_base[1]), pitch_eq


def _calibrate_leg(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> LegClosure:
    geometry = LEG_CLOSED_LOOP[side]
    motor_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, geometry.motor_joint)
    passive_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, geometry.passive_joints[0]
    )
    elbow_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, geometry.passive_joints[1]
    )
    wheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _wheel_joint_name(side))
    if min(motor_joint_id, passive_joint_id, elbow_joint_id, wheel_joint_id) < 0:
        raise ValueError(f"missing joint while calibrating {side} leg")

    eq_id = equality_id(model, geometry.equality_name)
    body1_id = int(model.eq_obj1id[eq_id])
    body2_id = int(model.eq_obj2id[eq_id])
    anchor1 = np.asarray(model.eq_data[eq_id, 0:3], dtype=float)
    anchor2 = np.asarray(model.eq_data[eq_id, 3:6], dtype=float)

    a_world = np.asarray(data.xanchor[motor_joint_id], dtype=float)
    e_world = np.asarray(data.xanchor[passive_joint_id], dtype=float)
    b_world = np.asarray(data.xanchor[elbow_joint_id], dtype=float)
    c_world = _body_local_point_world(data, body1_id, anchor1)
    w_world = np.asarray(data.xanchor[wheel_joint_id], dtype=float)

    axis = _unit(np.asarray(data.xaxis[motor_joint_id], dtype=float), f"{side} motor axis")
    ref = np.array([1.0, 0.0, 0.0])
    ex = ref - float(ref @ axis) * axis
    if np.linalg.norm(ex) < 1e-8:
        ref = np.array([0.0, 1.0, 0.0])
        ex = ref - float(ref @ axis) * axis
    ex = _unit(ex, f"{side} plane ex")
    ez = np.cross(axis, ex)

    def to2(point: np.ndarray) -> np.ndarray:
        delta = np.asarray(point, dtype=float) - a_world
        return np.array([float(delta @ ex), float(delta @ ez)])

    a = to2(a_world)
    e = to2(e_world)
    b = to2(b_world)
    c = to2(c_world)
    w = to2(w_world)

    r1 = float(np.linalg.norm(b - a))
    r2c = float(np.linalg.norm(c - b))
    r2w = float(np.linalg.norm(w - b))
    r3 = float(np.linalg.norm(c - e))
    if min(r1, r2c, r2w, r3) < 1e-9:
        raise ValueError(f"degenerate four-bar geometry on {side} leg")

    angle_ab = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
    angle_ec = float(np.arctan2(c[1] - e[1], c[0] - e[0]))
    angle_bc = float(np.arctan2(c[1] - b[1], c[0] - b[0]))
    angle_bw = float(np.arctan2(w[1] - b[1], w[0] - b[0]))

    motor_q = float(data.qpos[model.jnt_qposadr[motor_joint_id]])
    passive_q = float(data.qpos[model.jnt_qposadr[passive_joint_id]])
    elbow_q = float(data.qpos[model.jnt_qposadr[elbow_joint_id]])
    motor_offset = angle_ab - motor_q
    passive_offset = angle_ec - passive_q
    # In this processed MJCF, the elbow joint's positive qpos is opposite the
    # planar link2/link1 relative angle.
    elbow_offset = (angle_bc - angle_ab) + elbow_q

    branch_errors = []
    for branch_sign in (+1, -1):
        solved = _solve_closure(
            motor_q,
            closure_args=(a, e, r1, r2c, r3, motor_offset, branch_sign),
        )
        if solved is not None:
            c_candidate, _b_candidate = solved
            branch_errors.append((branch_sign, float(np.linalg.norm(c_candidate - c))))
    if not branch_errors:
        raise ValueError(f"cannot recover closure branch for {side} leg")

    return LegClosure(
        side=side,
        motor_joint_id=motor_joint_id,
        passive_joint_id=passive_joint_id,
        elbow_joint_id=elbow_joint_id,
        wheel_joint_id=wheel_joint_id,
        equality_body1_id=body1_id,
        equality_body2_id=body2_id,
        equality_anchor1=anchor1,
        equality_anchor2=anchor2,
        a_world=a_world,
        ex=ex,
        ez=ez,
        a=a,
        e=e,
        r1=r1,
        r2c=r2c,
        r2w=r2w,
        r3=r3,
        wbc_angle=_angle_wrap(angle_bw - angle_bc),
        motor_offset=float(motor_offset),
        passive_offset=float(passive_offset),
        elbow_offset=float(elbow_offset),
        branch_sign=min(branch_errors, key=lambda item: item[1])[0],
    )


def _solve_closure(
    theta_motor: float,
    *,
    closure_args: tuple[np.ndarray, np.ndarray, float, float, float, float, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    a, e, r1, r2c, r3, motor_offset, branch_sign = closure_args
    b = a + r1 * np.array([np.cos(theta_motor + motor_offset), np.sin(theta_motor + motor_offset)])
    eb = e - b
    distance = float(np.linalg.norm(eb))
    if distance > r2c + r3 + 1e-9 or distance < abs(r2c - r3) - 1e-9:
        return None
    along = (r2c * r2c - r3 * r3 + distance * distance) / (2.0 * distance)
    h_sq = r2c * r2c - along * along
    if h_sq < -1e-9:
        return None
    midpoint = b + along * eb / distance
    perpendicular = np.array([-eb[1], eb[0]]) / distance
    c = midpoint + branch_sign * np.sqrt(max(h_sq, 0.0)) * perpendicular
    return c, b


def _closure_args(closure: LegClosure, branch_sign: int | None = None):
    return (
        closure.a,
        closure.e,
        closure.r1,
        closure.r2c,
        closure.r3,
        closure.motor_offset,
        closure.branch_sign if branch_sign is None else branch_sign,
    )


def _apply_side_thetas(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    closures: dict[str, LegClosure],
    theta_motors: dict[str, float],
    previous_angles: dict[str, dict[str, float]],
) -> None:
    for side, closure in closures.items():
        theta_motor = float(theta_motors[side])
        solved = _solve_closure(theta_motor, closure_args=_closure_args(closure))
        if solved is None:
            raise ValueError(f"theta={theta_motor:.4f} is infeasible for {side} leg")
        c, b = solved
        angle_ab = float(np.arctan2(b[1] - closure.a[1], b[0] - closure.a[0]))
        angle_ec = float(np.arctan2(c[1] - closure.e[1], c[0] - closure.e[0]))
        angle_bc = float(np.arctan2(c[1] - b[1], c[0] - b[0]))
        passive = _nearest_equivalent_angle(
            angle_ec - closure.passive_offset,
            previous_angles[side]["passive"],
        )
        elbow = _nearest_equivalent_angle(
            closure.elbow_offset - (angle_bc - angle_ab),
            previous_angles[side]["elbow"],
        )
        previous_angles[side]["passive"] = passive
        previous_angles[side]["elbow"] = elbow

        data.qpos[model.jnt_qposadr[closure.motor_joint_id]] = theta_motor
        data.qpos[model.jnt_qposadr[closure.passive_joint_id]] = passive
        data.qpos[model.jnt_qposadr[closure.elbow_joint_id]] = elbow
        data.qpos[model.jnt_qposadr[closure.wheel_joint_id]] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _apply_theta(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    closures: dict[str, LegClosure],
    theta_motor: float,
    previous_angles: dict[str, dict[str, float]],
) -> None:
    _apply_side_thetas(
        model,
        data,
        closures,
        {side: float(theta_motor) for side in closures},
        previous_angles,
    )


def _connect_errors(data: mujoco.MjData, closures: dict[str, LegClosure]) -> dict[str, float]:
    return {
        side: float(
            np.linalg.norm(
                _body_local_point_world(data, closure.equality_body1_id, closure.equality_anchor1)
                - _body_local_point_world(data, closure.equality_body2_id, closure.equality_anchor2)
            )
        )
        for side, closure in closures.items()
    }


def _height_xipos(model: mujoco.MjModel, data: mujoco.MjData, base_id: int, wheel_ids: dict[str, int]) -> float:
    return float(
        data.xipos[base_id, 2]
        - 0.5 * (data.xipos[wheel_ids["left"], 2] + data.xipos[wheel_ids["right"], 2])
    )


def _side_heights_xipos(
    data: mujoco.MjData,
    base_id: int,
    wheel_ids: dict[str, int],
) -> dict[str, float]:
    base_z = float(data.xipos[base_id, 2])
    return {
        side: base_z - float(data.xipos[wheel_id, 2])
        for side, wheel_id in wheel_ids.items()
    }


def _anchor_height_from_closure(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    closures: dict[str, LegClosure],
    theta_motor: float,
    *,
    branch_sign: int | None = None,
) -> float:
    base_id = body_id(model, "base_link")
    heights = []
    for closure in closures.values():
        solved = _solve_closure(
            theta_motor,
            closure_args=_closure_args(closure, branch_sign),
        )
        if solved is None:
            raise ValueError(f"theta={theta_motor:.4f} is infeasible for {closure.side} leg")
        c, b = solved
        angle_bc = float(np.arctan2(c[1] - b[1], c[0] - b[0]))
        w = b + closure.r2w * np.array(
            [np.cos(angle_bc + closure.wbc_angle), np.sin(angle_bc + closure.wbc_angle)]
        )
        wheel_z = closure.a_world[2] + w[0] * closure.ex[2] + w[1] * closure.ez[2]
        heights.append(float(data.xpos[base_id, 2] - wheel_z))
    return float(np.mean(heights))


def _choose_low_height_branch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    closures: dict[str, LegClosure],
) -> dict[str, LegClosure]:
    selected = {}
    for side, closure in closures.items():
        candidate_heights = []
        for sign in (+1, -1):
            h = _anchor_height_from_closure(model, data, {side: closure}, THETA_SCAN_MIN, branch_sign=sign)
            candidate_heights.append((sign, h))
        selected_sign = min(candidate_heights, key=lambda item: item[1])[0]
        selected[side] = LegClosure(**{**closure.__dict__, "branch_sign": selected_sign})
    return selected


def _find_theta_for_height(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    closures: dict[str, LegClosure],
    base_id: int,
    wheel_ids: dict[str, int],
    target_h: float,
) -> float:
    previous_angles = {side: {"passive": 0.0, "elbow": 0.0} for side in closures}

    def height_at(theta: float) -> float:
        _apply_theta(model, data, closures, theta, previous_angles)
        return _height_xipos(model, data, base_id, wheel_ids)

    lo = THETA_SCAN_MIN
    hi = THETA_MAX
    if height_at(lo) > target_h or height_at(hi) < target_h:
        raise ValueError(f"target height {target_h:.4f} is outside analytic scan range")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if height_at(mid) < target_h:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _apply_named_qpos(model: mujoco.MjModel, data: mujoco.MjData, qpos_values: dict[str, float]) -> None:
    for joint_name, value in qpos_values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"missing joint in viewer-low qpos: {joint_name}")
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _current_passive_angles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    closures: dict[str, LegClosure],
) -> dict[str, dict[str, float]]:
    return {
        side: {
            "passive": float(data.qpos[model.jnt_qposadr[closure.passive_joint_id]]),
            "elbow": float(data.qpos[model.jnt_qposadr[closure.elbow_joint_id]]),
        }
        for side, closure in closures.items()
    }


def _sort_unique_by_height(
    heights: np.ndarray,
    values: np.ndarray,
    *,
    min_height_step: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(heights)
    h_sorted = heights[order]
    value_sorted = values[order]

    keep_idx = [0]
    for i in range(1, len(h_sorted)):
        if h_sorted[i] - h_sorted[keep_idx[-1]] > min_height_step:
            keep_idx.append(i)
    return h_sorted[keep_idx], value_sorted[keep_idx]


def _viewer_low_row(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    closures: dict[str, LegClosure],
    base_id: int,
    wheel_ids: dict[str, int],
) -> dict[str, float]:
    _apply_named_qpos(model, data, VIEWER_LOW_QPOS)
    errors = _connect_errors(data, closures)
    max_error = max(errors.values())
    if max_error > CONNECT_TOLERANCE:
        raise RuntimeError(
            "viewer-low connect anchor mismatch: "
            + ", ".join(f"{side}={err:.3e}" for side, err in errors.items())
        )
    h_xipos = _height_xipos(model, data, base_id, wheel_ids)
    dy_wheel, pitch_eq = _base_frame_geometry(model, data, base_id, wheel_ids)
    theta_left = float(data.qpos[model.jnt_qposadr[closures["left"].motor_joint_id]])
    theta_right = float(data.qpos[model.jnt_qposadr[closures["right"].motor_joint_id]])
    return {
        "theta_target": 0.5 * (theta_left + theta_right),
        "theta_actual_left": theta_left,
        "theta_actual_right": theta_right,
        "h": float(h_xipos),
        "h_anchor": float(h_xipos),
        "dy_wheel_in_base": float(dy_wheel),
        "pitch_eq": float(pitch_eq),
        "connect_error_left": errors["left"],
        "connect_error_right": errors["right"],
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    urdf_path = repo_root / "sim" / "robot" / "robot.urdf"
    cache_dir = repo_root / "tmp" / "probe_leg_geometry_analytic"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = prepare_controlled_mujoco_xml(urdf_path, cache_dir=cache_dir)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    addresses = model_addresses(model)

    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_id < 0:
        raise RuntimeError("missing 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)
    stand_root_qpos = data.qpos[addresses.root_qpos : addresses.root_qpos + 7].copy()
    stand_closures = {side: _calibrate_leg(model, data, side) for side in LEG_CLOSED_LOOP}
    stand_anchor_h = _anchor_height_from_closure(model, data, stand_closures, STAND_KEYFRAME_THETA)
    stand_wheel_joints = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _wheel_joint_name(side))
        for side in LEG_CLOSED_LOOP
    ]
    stand_mujoco_anchor_h = float(
        data.xpos[body_id(model, "base_link"), 2]
        - 0.5 * sum(float(data.xanchor[joint_id, 2]) for joint_id in stand_wheel_joints)
    )
    if abs(stand_anchor_h - stand_mujoco_anchor_h) > 1e-3:
        raise RuntimeError(
            f"stand anchor check failed: analytic={stand_anchor_h:.4f}, "
            f"mujoco={stand_mujoco_anchor_h:.4f}"
        )

    base_id = body_id(model, "base_link")
    wheel_ids = {side: body_id(model, geometry.wheel_body) for side, geometry in LEG_CLOSED_LOOP.items()}

    data.qpos[:] = 0.0
    data.qpos[addresses.root_qpos : addresses.root_qpos + 7] = stand_root_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    _apply_named_qpos(model, data, VIEWER_LOW_QPOS)
    closures = {side: _calibrate_leg(model, data, side) for side in LEG_CLOSED_LOOP}
    viewer_low = _viewer_low_row(model, data, closures, base_id, wheel_ids)

    theta_min = ANALYTIC_BRANCH_THETA_MIN
    theta_targets = np.concatenate(
        (
            np.array([theta_min]),
            np.arange(np.ceil(theta_min / THETA_STEP) * THETA_STEP, THETA_MAX + 0.5 * THETA_STEP, THETA_STEP),
        )
    )
    theta_targets = np.unique(np.round(theta_targets, 12))

    print(
        "viewer-low "
        f"theta_left={viewer_low['theta_actual_left']:.4f} "
        f"theta_right={viewer_low['theta_actual_right']:.4f} "
        f"h_xipos={viewer_low['h']:.5f} "
        f"err_left={viewer_low['connect_error_left']:.2e} "
        f"err_right={viewer_low['connect_error_right']:.2e}"
    )
    print(f"analytic side sweep theta=[{theta_targets[0]:.4f}, {theta_targets[-1]:.4f}] step={THETA_STEP:.4f}")
    print(
        f"{'theta':>8} {'h_left':>9} {'h_right':>9} {'h_mean':>9}"
        f" {'err_left':>10} {'err_right':>10}"
    )

    sample_rows = []
    previous_angles = _current_passive_angles(model, data, closures)
    for theta in theta_targets:
        _apply_theta(model, data, closures, float(theta), previous_angles)
        errors = _connect_errors(data, closures)
        max_error = max(errors.values())
        if max_error > CONNECT_TOLERANCE:
            raise RuntimeError(
                f"connect anchor mismatch at theta={theta:.4f}: "
                + ", ".join(f"{side}={err:.3e}" for side, err in errors.items())
            )
        side_heights = _side_heights_xipos(data, base_id, wheel_ids)
        h_xipos = _height_xipos(model, data, base_id, wheel_ids)
        h_anchor = _anchor_height_from_closure(model, data, closures, float(theta))
        dy_wheel, pitch_eq = _base_frame_geometry(model, data, base_id, wheel_ids)
        sample_rows.append(
            {
                "theta_target": float(theta),
                "theta_actual_left": float(theta),
                "theta_actual_right": float(theta),
                "h": float(h_xipos),
                "h_left": float(side_heights["left"]),
                "h_right": float(side_heights["right"]),
                "h_anchor": float(h_anchor),
                "dy_wheel_in_base": float(dy_wheel),
                "pitch_eq": float(pitch_eq),
                "connect_error_left": errors["left"],
                "connect_error_right": errors["right"],
            }
        )
        print(
            f"{theta:>8.4f} {side_heights['left']:>9.5f} {side_heights['right']:>9.5f}"
            f" {h_xipos:>9.5f}"
            f" {errors['left']:>10.2e} {errors['right']:>10.2e}"
        )

    sample_theta_left = np.array([row["theta_actual_left"] for row in sample_rows])
    sample_theta_right = np.array([row["theta_actual_right"] for row in sample_rows])
    sample_h_left = np.array([row["h_left"] for row in sample_rows])
    sample_h_right = np.array([row["h_right"] for row in sample_rows])
    h_left_sorted, theta_left_by_h = _sort_unique_by_height(sample_h_left, sample_theta_left)
    h_right_sorted, theta_right_by_h = _sort_unique_by_height(sample_h_right, sample_theta_right)

    h_min = max(float(h_left_sorted[0]), float(h_right_sorted[0]))
    h_max = min(float(h_left_sorted[-1]), float(h_right_sorted[-1]))
    if h_min >= h_max:
        raise RuntimeError(f"empty common LUT height range: h_min={h_min:.4f}, h_max={h_max:.4f}")

    h_targets = np.linspace(h_min, h_max, min(len(h_left_sorted), len(h_right_sorted)))
    theta_left_targets = np.interp(h_targets, h_left_sorted, theta_left_by_h)
    theta_right_targets = np.interp(h_targets, h_right_sorted, theta_right_by_h)

    rows = []
    previous_angles = _current_passive_angles(model, data, closures)
    for h_target, theta_left, theta_right in zip(h_targets, theta_left_targets, theta_right_targets):
        _apply_side_thetas(
            model,
            data,
            closures,
            {"left": float(theta_left), "right": float(theta_right)},
            previous_angles,
        )
        errors = _connect_errors(data, closures)
        max_error = max(errors.values())
        if max_error > CONNECT_TOLERANCE:
            raise RuntimeError(
                f"connect anchor mismatch at h={h_target:.4f}: "
                + ", ".join(f"{side}={err:.3e}" for side, err in errors.items())
            )
        side_heights = _side_heights_xipos(data, base_id, wheel_ids)
        dy_wheel, pitch_eq = _base_frame_geometry(model, data, base_id, wheel_ids)
        h_xipos = _height_xipos(model, data, base_id, wheel_ids)
        rows.append(
            {
                "h_target": float(h_target),
                "theta_target": float(0.5 * (theta_left + theta_right)),
                "theta_actual_left": float(theta_left),
                "theta_actual_right": float(theta_right),
                "h": float(h_xipos),
                "h_left": float(side_heights["left"]),
                "h_right": float(side_heights["right"]),
                "dy_wheel_in_base": float(dy_wheel),
                "pitch_eq": float(pitch_eq),
                "connect_error_left": errors["left"],
                "connect_error_right": errors["right"],
            }
        )

    theta_left_arr = np.array([row["theta_actual_left"] for row in rows])
    theta_right_arr = np.array([row["theta_actual_right"] for row in rows])
    theta_arr = 0.5 * (theta_left_arr + theta_right_arr)
    h_arr = np.array([row["h"] for row in rows])
    h_left_arr = np.array([row["h_left"] for row in rows])
    h_right_arr = np.array([row["h_right"] for row in rows])
    dy_arr = np.array([row["dy_wheel_in_base"] for row in rows])
    pitch_eq_arr = np.array([row["pitch_eq"] for row in rows])
    order = np.argsort(h_arr)
    theta_sorted = theta_arr[order]
    theta_left_sorted = theta_left_arr[order]
    theta_right_sorted = theta_right_arr[order]
    h_sorted = h_arr[order]
    h_left_grid = h_left_arr[order]
    h_right_grid = h_right_arr[order]
    dy_sorted = dy_arr[order]
    pitch_eq_sorted = pitch_eq_arr[order]

    diffs = np.diff(h_sorted)
    strictly_monotonic = bool(np.all(diffs > 0) or np.all(diffs < 0))
    sign_change_indices: list[int] = []
    if not strictly_monotonic:
        sign_change_indices = (np.where(np.diff(np.sign(diffs)) != 0)[0] + 1).tolist()
        raise RuntimeError(f"h(theta) is not strictly monotonic: {sign_change_indices}")
    output = {
        "theta_grid": theta_sorted.tolist(),
        "theta_left_grid": theta_left_sorted.tolist(),
        "theta_right_grid": theta_right_sorted.tolist(),
        "h_grid": h_sorted.tolist(),
        "h_left_grid": h_left_grid.tolist(),
        "h_right_grid": h_right_grid.tolist(),
        "dy_grid": dy_sorted.tolist(),
        "pitch_eq_grid": pitch_eq_sorted.tolist(),
        "h_min": float(h_sorted.min()),
        "h_max": float(h_sorted.max()),
        "theta_min": float(theta_sorted.min()),
        "theta_max": float(theta_sorted.max()),
        "monotonic": strictly_monotonic,
        "side_theta_monotonic": {
            "left": bool(np.all(np.diff(theta_left_sorted) > 0) or np.all(np.diff(theta_left_sorted) < 0)),
            "right": bool(np.all(np.diff(theta_right_sorted) > 0) or np.all(np.diff(theta_right_sorted) < 0)),
        },
        "sign_change_indices": sign_change_indices,
        "raw_same_theta_probe": sample_rows,
        "raw": rows,
    }

    lut_path = repo_root / "sim" / "controllers" / "leg_height_lut.json"
    with open(lut_path, "w") as f:
        json.dump(output, f, indent=2)

    lower_anchor_h = _anchor_height_from_closure(model, data, closures, ANALYTIC_BRANCH_THETA_MIN)
    dy_dh = np.gradient(dy_sorted, h_sorted)
    print()
    print(f"LUT saved to {lut_path}")
    print(f"  samples: {len(theta_sorted)}")
    print(f"  theta range: [{output['theta_min']:.4f}, {output['theta_max']:.4f}] rad")
    print(f"  h_xipos range: [{output['h_min']:.4f}, {output['h_max']:.4f}] m")
    print(
        f"  cmd_height lower bound {h_sorted.min():.4f} maps to "
        f"left={theta_left_sorted[0]:.4f}, right={theta_right_sorted[0]:.4f} rad"
    )
    print(
        f"  stand anchor cross-check: analytic h({STAND_KEYFRAME_THETA:.3f})="
        f"{stand_anchor_h:.4f} m, MuJoCo anchor h={stand_mujoco_anchor_h:.4f} m"
    )
    print(f"  analytic branch anchor h({ANALYTIC_BRANCH_THETA_MIN:.6f})={lower_anchor_h:.4f} m")
    print(f"  pitch_eq(h=0.142): {float(np.interp(0.142, h_sorted, pitch_eq_sorted)):.4f} rad")
    print(f"  dy_wheel_dh(h=0.142): {float(np.interp(0.142, h_sorted, dy_dh)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
