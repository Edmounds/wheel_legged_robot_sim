import argparse
from pathlib import Path
import numpy as np

from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
from sim.rollout import RolloutConfig, run_rollout
from sim.state import SimState

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-pitch", type=float, default=1000.0)
    parser.add_argument("--q-pitch-rate", type=float, default=100.0)
    parser.add_argument("--q-roll", type=float, default=1000.0)
    parser.add_argument("--q-roll-rate", type=float, default=100.0)
    parser.add_argument("--q-wheel-pos", type=float, default=10.0)
    parser.add_argument("--q-wheel-vel", type=float, default=100.0)
    parser.add_argument("--r-forward", type=float, default=2.0)
    parser.add_argument("--r-roll", type=float, default=2.0)
    parser.add_argument("--linear-x", type=float, default=0.0)
    parser.add_argument("--angular-z", type=float, default=0.0)
    parser.add_argument("--vel-ki", type=float, default=0.0)
    parser.add_argument("--yaw-ki", type=float, default=0.0)
    parser.add_argument("--pitch-lean-gain", type=float, default=0.02)
    parser.add_argument("--yaw-damping", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--torque-limit", type=float, default=10.0)
    parser.add_argument("--pitch-trim", type=float, default=0.0)
    args = parser.parse_args()

    # Modify parameters based on arguments
    from dataclasses import replace
    params = replace(STAND_PARAMS)
    params.pitch_trim = args.pitch_trim


    params.q_diag = np.array([
        args.q_pitch,
        args.q_pitch_rate,
        args.q_roll,
        args.q_roll_rate,
        args.q_wheel_pos,
        args.q_wheel_vel,
    ])
    params.r_diag = np.array([args.r_forward, args.r_roll])
    
    params.target_velocity = args.linear_x
    params.target_yaw_rate = args.angular_z
    params.velocity_ki = args.vel_ki
    params.yaw_ki = args.yaw_ki
    params.pitch_lean_gain = args.pitch_lean_gain
    params.yaw_damping = args.yaw_damping



    controller = CombinedController(params)

    # Wrap controller to record wheel torques
    ctrl_history = []
    def wrapped_controller(model, data, state: SimState):
        ctrl = controller(model, data, state)
        # assuming wheel joints are first two actuators, but let's look up index in state.py
        # Actually in combined.py wheel controls are returned in `data.ctrl` directly or mapped?
        # Wheel joints are usually 'left_wheel_joint', 'right_wheel_joint'. 
        # In sim.state.model_addresses it gives actuators mapping.
        # We will just save the whole ctrl array and parse it later.
        ctrl_history.append(ctrl.copy())
        return ctrl

    xml_path = Path("sim/robot/robot.urdf")
    config = RolloutConfig(duration=args.duration, scenario="stand")
    
    # Just to print initial state
    import mujoco
    from sim.model_xml import prepare_controlled_mujoco_xml
    from sim.state import extract_sim_state
    
    m_path = prepare_controlled_mujoco_xml(xml_path, cache_dir=config.cache_dir)
    m = mujoco.MjModel.from_xml_path(str(m_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    initial_state = extract_sim_state(m, d)
    print(f"Initial Pitch: {initial_state.pitch:.4f} rad")
    
    from sim.model_semantics import MODEL_SEMANTICS
    from sim.state import model_addresses
    addresses = model_addresses(m)
    left_idx = addresses.actuators[MODEL_SEMANTICS.wheel_joints[0]]
    print(f"Left wheel ctrlrange: {m.actuator_ctrlrange[left_idx]}")
    
    controller._initialize_lqr(m, d, initial_state)

    result = run_rollout(xml_path, config, wrapped_controller)

    # Analysis
    dx = result.metrics.forward_distance
    max_pitch = result.metrics.max_abs_pitch
    z_range = result.metrics.max_base_height - result.metrics.min_base_height
    failure = result.failure_reason

    # Parse torques
    from sim.model_semantics import MODEL_SEMANTICS
    from sim.state import model_addresses
    addresses = model_addresses(result.model)
    left_idx = addresses.actuators[MODEL_SEMANTICS.wheel_joints[0]]
    right_idx = addresses.actuators[MODEL_SEMANTICS.wheel_joints[1]]

    ctrl_arr = np.array(ctrl_history)
    if len(ctrl_arr) > 0:
        left_torque = ctrl_arr[:, left_idx]
        right_torque = ctrl_arr[:, right_idx]
        max_abs_left = np.max(np.abs(left_torque))
        max_abs_right = np.max(np.abs(right_torque))
        mean_diff_torque = np.mean(np.abs(left_torque - right_torque))
    else:
        max_abs_left = 0.0
        max_abs_right = 0.0
        mean_diff_torque = 0.0

    print(f"--- Results for duration {args.duration}s ---")
    print(f"Failure:       {failure}")
    print(f"Max Abs Pitch: {max_pitch:.4f} rad")
    print(f"Forward Dist:  {dx:.4f} m")
    print(f"Z Range:       {z_range:.4f} m")
    print(f"Sat Ratio:     {result.metrics.saturation_ratio:.4f}")
    print(f"Max Abs L/R:   {max_abs_left:.4f} / {max_abs_right:.4f}, Mean Diff={mean_diff_torque:.4f}")

if __name__ == "__main__":
    main()
