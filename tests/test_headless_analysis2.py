import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import math
import numpy as np
import tempfile
import xml.etree.ElementTree as ET

from sim.mujoco_mesh_preprocess import prepare_mujoco_xml
from sim.model_xml import _urdf_to_mjcf, _ensure_equality_constraints, _configure_geom_collisions

urdf_path = Path("sim/robot/robot.urdf").resolve()
tmp_dir = Path(tempfile.mkdtemp())
converted_xml = _urdf_to_mjcf(urdf_path, tmp_dir)
prepared_xml = prepare_mujoco_xml(converted_xml)
tree = ET.parse(prepared_xml)
root = tree.getroot()

LEG_MOTOR_JOINTS = ("base_link_旋转-2", "base_link_旋转-1")
actuator = ET.SubElement(root, "actuator")
for jname in LEG_MOTOR_JOINTS:
    ET.SubElement(actuator, "position", {
        "name": f"act_{jname}",
        "joint": jname,
        "kp": "5",
        "ctrllimited": "true",
        "ctrlrange": "-1.5 1.5",
        "gear": "1.0",
    })

option = root.find("option")
if option is None:
    option = ET.Element("option", {"gravity": "0 0 0"})
    root.insert(0, option)
else:
    option.set("gravity", "0 0 0")

_configure_geom_collisions(root)
_ensure_equality_constraints(root)

out_xml = tmp_dir / "check_model.xml"
tree.write(out_xml, encoding="utf-8", xml_declaration=False)

model = mujoco.MjModel.from_xml_path(str(out_xml))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

joints_left = ["base_link_旋转-2", "base_link_旋转-4", "link1_left_旋转-6", "link2_left_旋转-13"]
joints_right = ["base_link_旋转-1", "base_link_旋转-3", "link1_right_旋转-5", "link2_right_旋转-12"]

# Get ids
j_ids_l = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints_left]
j_ids_r = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints_right]

qpos_adr_l = [model.jnt_qposadr[jid] for jid in j_ids_l]
qpos_adr_r = [model.jnt_qposadr[jid] for jid in j_ids_r]

history_l = []
history_r = []

timestep = model.opt.timestep
steps = int(6.28 / timestep)

for step in range(steps):
    t = step * timestep
    ctrl_val = math.sin(t)
    for act_id in range(model.nu):
        data.ctrl[act_id] = ctrl_val
        
    mujoco.mj_step(model, data)
    
    history_l.append([data.qpos[adr] for adr in qpos_adr_l])
    history_r.append([data.qpos[adr] for adr in qpos_adr_r])

hl = np.array(history_l)
hr = np.array(history_r)

print("WITH GEAR=1.0 for BOTH:")
print("Max values Left: ", np.max(hl, axis=0))
print("Max values Right:", np.max(hr, axis=0))
print("Min values Left: ", np.min(hl, axis=0))
print("Min values Right:", np.min(hr, axis=0))

range_l = np.max(hl, axis=0) - np.min(hl, axis=0)
range_r = np.max(hr, axis=0) - np.min(hr, axis=0)
print("\nLeft vs Right Discrepancy %:", np.abs(range_l - range_r) / range_l * 100)
