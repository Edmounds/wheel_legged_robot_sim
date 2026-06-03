#!/usr/bin/env python3
"""
Extracts structural and kinematic information from the processed MuJoCo XML
and saves it to docs/ROBOT_STRUCTURE.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure sim module is accessible
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import mujoco
import numpy as np

from sim.model_xml import prepare_controlled_mujoco_xml


def compute_mesh_local_bounding_box(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute local bounding box (min, max, size, center) for a given mesh id."""
    vert_start = model.mesh_vertadr[mesh_id]
    vert_num = model.mesh_vertnum[mesh_id]
    if vert_num == 0:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    verts = model.mesh_vert[vert_start : vert_start + vert_num]
    min_bound = verts.min(axis=0)
    max_bound = verts.max(axis=0)
    size = max_bound - min_bound
    center = (max_bound + min_bound) / 2.0
    return min_bound, max_bound, size, center


def generate_markdown() -> str:
    model_path = project_root / "sim" / "robot" / "robot.urdf"
    if not model_path.is_file():
        return f"Error: {model_path} not found."

    # Use the preprocessing pipeline to get the true simulation model
    prepared_xml = prepare_controlled_mujoco_xml(model_path)
    model = mujoco.MjModel.from_xml_path(str(prepared_xml))
    data = mujoco.MjData(model)

    stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_key_id < 0:
        raise RuntimeError("processed model is missing the 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand_key_id)
    mujoco.mj_forward(model, data)

    lines = [
        "# Robot Structure and Kinematics",
        "",
        "> **IMPORTANT:** This document represents the robot state after `sim/robot/robot.urdf` is converted and postprocessed into MJCF, then reset to the `stand` keyframe. Use this processed model for simulation structure checks, not the raw URDF alone.",
        "",
        "## Bodies and Kinematic Tree",
        ""
    ]

    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue

        parent_id = model.body_parentid[body_id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) if parent_id != 0 else "world"

        mass = model.body_mass[body_id]
        global_pos = data.xpos[body_id]
        global_quat = data.xquat[body_id]

        lines.append(f"### Body: `{body_name}`")
        lines.append(f"- **Parent**: `{parent_name}`")
        lines.append(f"- **Mass**: {mass:.4f} kg")
        lines.append(f"- **Global Pos (standing)**: [{global_pos[0]:.4f}, {global_pos[1]:.4f}, {global_pos[2]:.4f}]")
        lines.append(f"- **Global Quat (w,x,y,z)**: [{global_quat[0]:.4f}, {global_quat[1]:.4f}, {global_quat[2]:.4f}, {global_quat[3]:.4f}]")

        # Joints in this body
        jnt_num = model.body_jntnum[body_id]
        jnt_adr = model.body_jntadr[body_id]
        if jnt_num > 0:
            lines.append("- **Joints**:")
            for j in range(jnt_adr, jnt_adr + jnt_num):
                jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                jnt_type = model.jnt_type[j]
                axis = model.jnt_axis[j]
                type_str = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jnt_type, "unknown")

                limit_str = "unlimited"
                if model.jnt_limited[j]:
                    limit_str = f"[{model.jnt_range[j, 0]:.4f}, {model.jnt_range[j, 1]:.4f}]"

                lines.append(f"  - `{jnt_name}` ({type_str}) - Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}] - Limits: {limit_str}")

        # Geoms and Bounding Boxes
        geom_num = model.body_geomnum[body_id]
        geom_adr = model.body_geomadr[body_id]
        if geom_num > 0:
            lines.append("- **Geoms**:")
            for g in range(geom_adr, geom_adr + geom_num):
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
                geom_type = model.geom_type[g]
                is_mesh = geom_type == mujoco.mjtGeom.mjGEOM_MESH

                if is_mesh:
                    mesh_id = model.geom_dataid[g]
                    min_b, max_b, size, center = compute_mesh_local_bounding_box(model, mesh_id)
                    lines.append(f"  - `{geom_name}` (Mesh) - **Local Bounding Box Size (X, Y, Z)**: [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}]")
                    lines.append(f"    - Center offset: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
                else:
                    g_size = model.geom_size[g]
                    type_str = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}.get(geom_type, str(geom_type))
                    lines.append(f"  - `{geom_name}` ({type_str}) - Size parameters: [{g_size[0]:.4f}, {g_size[1]:.4f}, {g_size[2]:.4f}]")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    out_path = docs_dir / "ROBOT_STRUCTURE.md"

    content = generate_markdown()
    out_path.write_text(content, encoding="utf-8")
    print(f"Robot structure documentation written to {out_path}")
