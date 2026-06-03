"""Self-verify the right-hip axis flip in robot.urdf.

After flipping the right hip motor axis, the goal is:
- Motor reaction torques on base from same-direction ctrl should cancel (no pitch).
- A "mirrored" ctrl (left=+c, right=-c) should produce a symmetric leg extension,
  matching the original kinematic convention only through the dual-grid LUT.
- Cross-checks on left-only vs right-only ctrl should show mirrored angular
  accelerations: pitch matches, roll opposite.

We run mj_forward with various ctrl combinations on the hip motors and read
the base angular acceleration. Wheels and command sliders are held at zero.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from sim.model_semantics import MODEL_SEMANTICS
from sim.model_xml import prepare_controlled_mujoco_xml
from sim.state import model_addresses


URDF_PATH = Path(__file__).resolve().parents[1] / "sim" / "robot" / "robot.urdf"

LEFT_MOTOR = "base_link_旋转-2"
RIGHT_MOTOR = "base_link_旋转-1"


def _build_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    xml_path = prepare_controlled_mujoco_xml(URDF_PATH)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def _reset_to_mirrored_standing(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set qpos to a mirrored standing pose accounting for the right-axis flip.

    Original keyframe assumes both motors at +0.752. After flipping the right
    hip axis, the right motor's qpos must be negated to represent the same
    physical extension.
    """
    mujoco.mj_resetData(model, data)
    addrs = model_addresses(model)

    # Base at z so wheels are on the ground.
    data.qpos[addrs.root_qpos + 0] = 0.0
    data.qpos[addrs.root_qpos + 1] = 0.0
    data.qpos[addrs.root_qpos + 2] = 0.175
    data.qpos[addrs.root_qpos + 3 : addrs.root_qpos + 7] = [1, 0, 0, 0]

    # Active motors: mirrored sign convention.
    data.qpos[addrs.joint_qpos[LEFT_MOTOR]] = 0.752
    data.qpos[addrs.joint_qpos[RIGHT_MOTOR]] = -0.752

    # Passive knees (link1_*_旋转-5/6). Axes in URDF:
    #   left knee 旋转-6 axis = (-, +Y, -)
    #   right knee 旋转-5 axis = (-, +Y, -) (currently identical to left)
    # Backup convention had right knee with -Y axis. We mirror the passive
    # qpos in line with the configured axis signs at runtime.
    data.qpos[addrs.joint_qpos["link1_left_旋转-6"]] = 0.980
    data.qpos[addrs.joint_qpos["link1_right_旋转-5"]] = 0.980

    # Base idlers (link3_left/right): left=-Y axis, right=+Y axis already mirrored.
    data.qpos[addrs.joint_qpos["base_link_旋转-4"]] = 0.720  # left
    data.qpos[addrs.joint_qpos["base_link_旋转-3"]] = -0.720  # right (mirrored sign)

    # Wheels: zero.
    data.qpos[addrs.joint_qpos["link2_left_旋转-13"]] = 0.0
    data.qpos[addrs.joint_qpos["link2_right_旋转-12"]] = 0.0

    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0


def _settle_constraints(model: mujoco.MjModel, data: mujoco.MjData, n_steps: int = 200) -> None:
    """Allow the equality constraints to re-converge before we measure."""
    addrs = model_addresses(model)
    # Hold all motor ctrls at zero, let constraints work it out.
    for _ in range(n_steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

    # After settling, take note of the resulting motor angles.
    actual_left = float(data.qpos[addrs.joint_qpos[LEFT_MOTOR]])
    actual_right = float(data.qpos[addrs.joint_qpos[RIGHT_MOTOR]])
    print(f"  settled motor angles  left={actual_left:+.4f}  right={actual_right:+.4f}")


def _apply_ctrl_and_read_qacc(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl_left: float,
    ctrl_right: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Set just the hip motor ctrls, call mj_forward, return base linear and angular qacc."""
    addrs = model_addresses(model)
    data.ctrl[:] = 0.0
    data.ctrl[addrs.actuators[LEFT_MOTOR]] = ctrl_left
    data.ctrl[addrs.actuators[RIGHT_MOTOR]] = ctrl_right
    mujoco.mj_forward(model, data)
    qacc = np.array(data.qacc[addrs.root_qvel : addrs.root_qvel + 6])
    return qacc[:3], qacc[3:]


def main() -> None:
    print("Building model from URDF…")
    model, data = _build_model()

    print("Resetting to mirrored standing pose …")
    _reset_to_mirrored_standing(model, data)
    print("Settling equality constraints …")
    _settle_constraints(model, data)

    # Freeze qpos/qvel at the settled value for repeated mj_forward calls.
    qpos_settled = data.qpos.copy()
    qvel_settled = data.qvel.copy()

    def at_settled() -> None:
        data.qpos[:] = qpos_settled
        data.qvel[:] = qvel_settled

    at_settled()
    base_lin, base_ang = _apply_ctrl_and_read_qacc(model, data, 0.0, 0.0)
    print()
    print("Baseline (ctrl=0,0):")
    print(f"  base lin qacc = {base_lin}")
    print(f"  base ang qacc = {base_ang}")
    print()

    # mujoco qvel order for freejoint: [vx, vy, vz, wx, wy, wz] in WORLD frame.
    # state.pitch_rate = -wx, roll_rate = +wy (from sim/state.py). So:
    #   ang[0] (wx) ↔ -pitch dir
    #   ang[1] (wy) ↔ +roll dir
    #   ang[2] (wz) ↔ yaw
    cases = [
        ("ctrl=[+1, +1]  (same direction)",  +1.0, +1.0),
        ("ctrl=[-1, -1]  (same direction)",  -1.0, -1.0),
        ("ctrl=[+1, -1]  (anti direction)",  +1.0, -1.0),
        ("ctrl=[-1, +1]  (anti direction)",  -1.0, +1.0),
        ("ctrl=[+1,  0]  (left only)",       +1.0,  0.0),
        ("ctrl=[ 0, +1]  (right only)",       0.0, +1.0),
        ("ctrl=[-1,  0]  (left only neg)",   -1.0,  0.0),
        ("ctrl=[ 0, -1]  (right only neg)",   0.0, -1.0),
    ]

    rows = []
    for label, cl, cr in cases:
        at_settled()
        lin, ang = _apply_ctrl_and_read_qacc(model, data, cl, cr)
        delta_ang = ang - base_ang
        delta_lin = lin - base_lin
        rows.append((label, cl, cr, delta_lin, delta_ang))
        print(f"{label}")
        print(f"  Δ base lin qacc = {delta_lin}")
        print(f"  Δ base ang qacc = {delta_ang}")
        print(f"     → wx(-pitch) = {delta_ang[0]:+.4f}   wy(roll) = {delta_ang[1]:+.4f}   wz(yaw) = {delta_ang[2]:+.4f}")
        print()

    # Symmetry checks (expected after a correct mirror flip):
    print("=" * 60)
    print("Symmetry checks (after right-hip axis flip):")
    print("=" * 60)

    same_pp = next(r for r in rows if r[0].startswith("ctrl=[+1, +1]"))
    anti_pn = next(r for r in rows if r[0].startswith("ctrl=[+1, -1]"))
    left_only = next(r for r in rows if r[0].startswith("ctrl=[+1,  0]"))
    right_only = next(r for r in rows if r[0].startswith("ctrl=[ 0, +1]"))

    # (a) Same-direction ctrl should not produce direct pitch reaction.
    # Note: settled-state asymmetry may leak in via constraints, so we check
    # the magnitude relative to anti-direction case.
    pp_pitch = abs(same_pp[4][0])
    pn_pitch = abs(anti_pn[4][0])
    print(f"  |pitch from ctrl=[+1,+1]| = {pp_pitch:.4f}")
    print(f"  |pitch from ctrl=[+1,-1]| = {pn_pitch:.4f}")
    print(f"    expect pp << pn after the flip (pp ≈ 0).")

    pp_roll = abs(same_pp[4][1])
    pn_roll = abs(anti_pn[4][1])
    print(f"  |roll from ctrl=[+1,+1]| = {pp_roll:.4f}")
    print(f"  |roll from ctrl=[+1,-1]| = {pn_roll:.4f}")
    print(f"    expect pp > pn after the flip (roll motion now from same-ctrl).")

    # (b) Mirrored single-leg ctrl: left=+1 alone vs right=-1 alone should give
    # mirror-image angular accel (same pitch sign, opposite roll sign).
    left_p, left_r = left_only[4][0], left_only[4][1]
    right_neg = next(r for r in rows if r[0].startswith("ctrl=[ 0, -1]"))
    rneg_p, rneg_r = right_neg[4][0], right_neg[4][1]
    print()
    print(f"  Δ from ctrl=[+1, 0]   : pitch(wx)={left_p:+.4f}  roll(wy)={left_r:+.4f}")
    print(f"  Δ from ctrl=[ 0,-1]   : pitch(wx)={rneg_p:+.4f}  roll(wy)={rneg_r:+.4f}")
    print(f"    mirror expectation: pitch matches, roll opposite signs.")

    # (c) Right-only ctrl[+1] should mirror left-only ctrl[+1] before the flip;
    # AFTER the flip, right-only ctrl[+1] is the OPPOSITE direction physically.
    right_pos = right_only[4]
    print()
    print(f"  Δ from ctrl=[+1, 0]: ang = {left_only[4]}")
    print(f"  Δ from ctrl=[ 0,+1]: ang = {right_pos}")
    print(f"    after flip: these should be OPPOSITE signs in BOTH pitch and roll")
    print(f"    (because right's +ctrl now rotates link1 the opposite physical way).")


if __name__ == "__main__":
    main()
