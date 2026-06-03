import mujoco
from pathlib import Path
from sim.rollout import RolloutConfig, run_rollout
from sim.controllers.combined import CombinedController
from sim.controllers.default_params import STAND_PARAMS
import numpy as np
import cv2

xml_path = Path("sim/robot/robot.urdf")
from sim.model_xml import prepare_controlled_mujoco_xml
m_path = prepare_controlled_mujoco_xml(xml_path, cache_dir=Path(".cache"))
m = mujoco.MjModel.from_xml_path(str(m_path))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

params = STAND_PARAMS
params.q_diag = np.array([500.0, 50.0, 500.0, 50.0, 10.0, 1.0])
controller = CombinedController(params)
controller._initialize_lqr(m, d, __import__('sim.state').state.extract_sim_state(m, d))

renderer = mujoco.Renderer(m, 480, 640)
frames = []

for i in range(int(5.0 / m.opt.timestep)):
    state = __import__('sim.state').state.extract_sim_state(m, d)
    ctrl = controller(m, d, state)
    d.ctrl[:] = ctrl
    mujoco.mj_step(m, d)
    
    if i % 20 == 0:
        renderer.update_scene(data=d, camera="camera1") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "camera1") != -1 else renderer.update_scene(data=d)
        pixels = renderer.render()
        frames.append(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

out = cv2.VideoWriter('artifacts/success_balance.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 100, (640, 480))
for frame in frames:
    out.write(frame)
out.release()
print("Saved artifacts/success_balance.mp4")
