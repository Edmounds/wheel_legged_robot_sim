import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import mujoco
import mediapy as media

from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
from sim.rollout import RolloutConfig
from sim.state import SimState, extract_sim_state
from sim.model_xml import prepare_controlled_mujoco_xml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--q-pitch", type=float, default=1000.0)
    parser.add_argument("--q-pitch-rate", type=float, default=100.0)
    parser.add_argument("--q-roll", type=float, default=1000.0)
    parser.add_argument("--q-roll-rate", type=float, default=100.0)
    parser.add_argument("--q-wheel-pos", type=float, default=10.0)
    parser.add_argument("--q-wheel-vel", type=float, default=1.0)
    parser.add_argument("--r-forward", type=float, default=2.0)
    parser.add_argument("--r-roll", type=float, default=2.0)
    args = parser.parse_args()

    # Modify parameters based on arguments
    params = replace(STAND_PARAMS, vmc=replace(STAND_PARAMS.vmc))

    params.q_diag = np.array([
        args.q_pitch,
        args.q_pitch_rate,
        args.q_roll,
        args.q_roll_rate,
        args.q_wheel_pos,
        args.q_wheel_vel,
    ])
    params.r_diag = np.array([args.r_forward, args.r_roll])
    
    controller = CombinedController(params)

    xml_path = Path("sim/robot/robot.urdf")
    m_path = prepare_controlled_mujoco_xml(xml_path)
    m = mujoco.MjModel.from_xml_path(str(m_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    renderer = mujoco.Renderer(m, 480, 640)
    frames = []

    controller._initialize_lqr(m, d, extract_sim_state(m, d))

    fps = 30
    steps_per_frame = int((1.0 / fps) / m.opt.timestep)

    print("Running simulation and recording...")
    step_count = int(args.duration / m.opt.timestep)
    for i in range(step_count):
        state = extract_sim_state(m, d)
        ctrl = controller(m, d, state)
        d.ctrl[:] = np.clip(ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
        
        if i % steps_per_frame == 0:
            renderer.update_scene(d, camera=-1)
            frames.append(renderer.render())
            
        if abs(state.pitch) > 1.5:
            print(f"Fell at step {i}")
            break

    out_path = "outputs/lqr_test.mp4"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    media.write_video(out_path, frames, fps=fps)
    print(f"Video saved to {out_path}")

if __name__ == "__main__":
    main()
