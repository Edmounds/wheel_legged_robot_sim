#!/usr/bin/env python3
"""
Launch the robot URDF via MuJoCo's built-in converter (base fixed, no freejoint).
Adds position actuators on leg motors for manual slider control.
Use to verify model integrity (linkage, meshes, constraints).
"""

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import mujoco.viewer

from src.mujoco_mesh_preprocess import prepare_mujoco_xml

LEG_MOTOR_JOINTS = ("base_link_旋转-2", "base_link_旋转-1")


def main():
    parser = argparse.ArgumentParser(description="Launch MuJoCo viewer for URDF testing.")
    parser.add_argument("--urdf", type=Path, default=Path("src/robot/robot.urdf"))
    args = parser.parse_args()

    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        print(f"Error: URDF file {urdf_path} does not exist.")
        sys.exit(1)

    # Load URDF via MuJoCo (handles oversized mesh splitting) and preprocess
    from src.mjcf_builder import _urdf_to_mjcf, _ensure_equality_constraints, _configure_geom_collisions, _apply_link_materials
    tmp_dir = Path(tempfile.mkdtemp())
    converted_xml = _urdf_to_mjcf(urdf_path, tmp_dir)

    prepared_xml = prepare_mujoco_xml(converted_xml)
    prepared_dir = Path(prepared_xml).parent

    tree = ET.parse(prepared_xml)
    root = tree.getroot()

    # Make mesh paths absolute so temp output dir can still find them
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
        
    for mesh in asset.findall("mesh"):
        f = mesh.get("file")
        if f and not Path(f).is_absolute():
            mesh.set("file", str((prepared_dir / f).resolve()))

    # Add environment assets (checkerboard floor and skybox)
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

    # Add environment into worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "pos": "0 0 -1.0", 
        "size": "0 0 1", "material": "grid"
    })
    ET.SubElement(worldbody, "light", {
        "pos": "0 0 3", "dir": "0 0 -1", "directional": "true", "castshadow": "false"
    })

    # Visual settings
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"ambient": ".1 .1 .1", "diffuse": ".6 .6 .6", "specular": ".3 .3 .3"})
    ET.SubElement(visual, "rgba", {"haze": ".15 .25 .35 1"})
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

    # Position actuators for direct angle control.
    # With gravity off, slider value = target angle (no external forces).
    actuator = ET.SubElement(root, "actuator")
    for jname in LEG_MOTOR_JOINTS:
        ET.SubElement(actuator, "position", {
            "name": f"act_{jname}",
            "joint": jname,
            "kp": "50",
            "ctrllimited": "true",
            "ctrlrange": "-1.5 1.5",
            "gear": "1.0",
        })

    # Disable gravity so the unconstrained passive links don't fall down
    option = root.find("option")
    if option is None:
        option = ET.Element("option", {"gravity": "0 0 0"})
        root.insert(0, option)
    else:
        option.set("gravity", "0 0 0")

    # Disable mesh collisions (avoid internal tension from non-convex CAD overlaps)
    _configure_geom_collisions(root)

    # Tint legs (carbon fibre) and link3 (aluminium) so the model is not uniform grey.
    _apply_link_materials(root)

    # Add the virtual connections (equality constraints).
    _ensure_equality_constraints(root)

    out_xml = Path(tempfile.mkdtemp()) / "check_model.xml"
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)

    model = mujoco.MjModel.from_xml_path(str(out_xml))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    # Sync ctrl sliders to current joint positions so legs don't jump
    for act_id in range(model.nu):
        if model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
            j_id = model.actuator_trnid[act_id, 0]
            gear = model.actuator_gear[act_id, 0]
            if gear != 0:
                data.ctrl[act_id] = data.qpos[model.jnt_qposadr[j_id]] / gear

    import time
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- 默认开启 base_link 相机跟随 ---
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        print("=" * 60)
        print("Base is FIXED. Gravity is OFF.")
        print("Drag Control sliders to set joint angles directly.")
        print(f"  act_{LEG_MOTOR_JOINTS[0]}  ->  left leg")
        print(f"  act_{LEG_MOTOR_JOINTS[1]}  ->  right leg")
        print("Slider value = target angle (radians).")
        print("=" * 60)

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(float(model.opt.timestep))


if __name__ == "__main__":
    main()
