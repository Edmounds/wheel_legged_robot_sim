#!/usr/bin/env python3
"""
Test script to programmatically drive both leg actuators with a synchronized 
sine wave to visually observe any kinematic asymmetry in the viewer.
"""

import argparse
import math
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer

from sim.mujoco_mesh_preprocess import prepare_mujoco_xml

LEG_MOTOR_JOINTS = ("base_link_rot-2", "base_link_rot-1") # Note: assuming renamed to rot as per rules, will fallback to 旋转 if needed
LEG_MOTOR_JOINTS_LEGACY = ("base_link_旋转-2", "base_link_旋转-1")

def main():
    parser = argparse.ArgumentParser(description="Test kinematic symmetry by driving actuators simultaneously.")
    parser.add_argument("--urdf", type=Path, default=Path("sim/robot/robot.urdf"))
    args = parser.parse_args()

    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        print(f"Error: URDF file {urdf_path} does not exist.")
        sys.exit(1)

    # Load URDF via MuJoCo and preprocess
    from sim.model_xml import _urdf_to_mjcf, _ensure_equality_constraints, _configure_geom_collisions
    tmp_dir = Path(tempfile.mkdtemp())
    converted_xml = _urdf_to_mjcf(urdf_path, tmp_dir)

    prepared_xml = prepare_mujoco_xml(converted_xml)
    prepared_dir = Path(prepared_xml).parent

    tree = ET.parse(prepared_xml)
    root = tree.getroot()

    # Make mesh paths absolute
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
        
    for mesh in asset.findall("mesh"):
        f = mesh.get("file")
        if f and not Path(f).is_absolute():
            mesh.set("file", str((prepared_dir / f).resolve()))

    # Add environment assets
    ET.SubElement(asset, "texture", {
        "name": "skybox", "type": "skybox", "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "512"
    })
    ET.SubElement(asset, "texture", {
        "name": "checker", "type": "2d", "builtin": "checker",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "width": "512", "height": "512",
        "mark": "cross", "markrgb": ".8 .8 .8"
    })
    ET.SubElement(asset, "material", {
        "name": "grid", "texture": "checker", "texrepeat": "1 1",
        "texuniform": "true", "reflectance": "0.2"
    })

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "pos": "0 0 -1.0", 
        "size": "0 0 1", "material": "grid"
    })
    ET.SubElement(worldbody, "light", {
        "pos": "0 0 3", "dir": "0 0 -1", "directional": "true", "castshadow": "true"
    })

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"ambient": ".1 .1 .1", "diffuse": ".6 .6 .6", "specular": ".3 .3 .3"})
    ET.SubElement(visual, "rgba", {"haze": ".15 .25 .35 1"})
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

    # Check which joint names exist in the URDF to be safe
    urdf_content = urdf_path.read_text(encoding="utf-8")
    motor_joints = LEG_MOTOR_JOINTS if "rot-1" in urdf_content else LEG_MOTOR_JOINTS_LEGACY

    actuator = ET.SubElement(root, "actuator")
    for jname in motor_joints:
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

    out_xml = Path(tempfile.mkdtemp()) / "check_model.xml"
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)

    model = mujoco.MjModel.from_xml_path(str(out_xml))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("=" * 60)
        print("Running symmetry test: Both legs driven by a synchronized sine wave.")
        print("Watch the movement speeds and joint poses in the viewer.")
        print("=" * 60)
        
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            
            # Generate a slow sine wave: amplitude 1.0, period ~6.28 seconds
            ctrl_val = math.sin(t)
            
            # Apply exactly the same control signal to both leg actuators
            for act_id in range(model.nu):
                data.ctrl[act_id] = ctrl_val
                
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(float(model.opt.timestep))

if __name__ == "__main__":
    main()
