from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import atan2, cos, sin
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from src.model_semantics import MODEL_SEMANTICS, WHEEL_RADIUS
from src.mujoco_mesh_preprocess import prepare_mujoco_xml


CMD_SLIDER_NAMES = ("cmd_linear_x", "cmd_angular_z", "cmd_height", "cmd_jump")


@dataclass(frozen=True)
class SingleWheelTrapezoidTerrain:
    side: str = "left"
    height: float = 0.065
    ramp_length: float = 0.20
    platform_length: float = 0.25
    width: float = 0.40
    y_start: float = 0.22
    side_x: float = 0.125


@dataclass(frozen=True)
class WavyRoadTerrain:
    """Washboard road along y (forward): sinusoid with per-crest random height, constant across x."""

    y_start: float = 1.05
    length: float = 1.20          # along y (forward direction)
    width: float = 0.60           # along x (lateral)
    amplitude: float = 0.03       # half peak-to-peak; max peak-to-peak = 60 mm before random scaling
    wavelength: float = 0.35      # along y
    base_depth: float = 0.02
    height_min_scale: float = 0.6  # each crest height ~ uniform[min_scale, 1] * 2*amplitude (>= old 36 mm)
    seed: int = 0                 # RNG seed for per-crest height randomisation
    nrow: int = 241               # along y: ~5 mm/cell, ~70 cells per wavelength
    ncol: int = 13                # along x: ~50 mm/cell, sufficient (profile constant in x)


def _urdf_to_mjcf(urdf_path: Path, output_dir: Path) -> Path:
    """Convert a URDF file to MJCF using MuJoCo's built-in converter."""
    urdf_tree = ET.parse(urdf_path)
    urdf_root = urdf_tree.getroot()

    # Absolutize mesh paths so the temp URDF can find them
    for mesh_el in urdf_root.iter("mesh"):
        filename = mesh_el.get("filename")
        if not filename:
            continue
        mesh_path = (urdf_path.parent / filename).resolve()
        if mesh_path.is_file():
            mesh_el.set("filename", str(mesh_path))

    tmp_urdf = output_dir / urdf_path.name
    urdf_tree.write(tmp_urdf, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
    mjcf_path = output_dir / f"{urdf_path.stem}_converted.xml"
    mujoco.mj_saveLastXML(str(mjcf_path), model)
    return mjcf_path


def prepare_controlled_mujoco_xml(
    urdf_path: Path,
    cache_dir: Path | None = None,
    terrain: str | None = None,
    terrain_side: str = "left",
) -> Path:
    """Create a controlled MJCF model from the URDF simulation source.

    Accepts a URDF file, converts it to MJCF via MuJoCo's built-in
    converter, then applies post-processing (freejoint, actuators,
    equality constraints, etc.).
    """
    urdf_path = urdf_path.expanduser().resolve()
    output_root = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else Path(tempfile.mkdtemp(prefix="mujoco_controlled_"))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # Step 1: URDF -> MJCF via MuJoCo
    converted_xml = _urdf_to_mjcf(urdf_path, output_root)

    # Step 2: Mesh preprocessing on the converted MJCF
    mesh_cache_dir = output_root / "mesh_preprocess"
    prepared_xml = prepare_mujoco_xml(converted_xml, cache_dir=mesh_cache_dir)
    prepared_xml = prepared_xml.expanduser().resolve()

    tree = ET.parse(prepared_xml)
    root = tree.getroot()

    _make_mesh_paths_absolute(root, prepared_xml.parent)
    _ensure_world_environment(root)
    _apply_link_materials(root)
    _ensure_test_terrain(root, terrain, terrain_side, output_root)
    _ensure_root_freejoint(root)
    _ensure_command_slider_joints(root)
    _configure_geom_collisions(root)
    _add_collision_proxies(root)
    _replace_actuators(root)
    _ensure_equality_constraints(root)
    _stiffen_leg_motor_joint_limits(root)
    _ensure_standing_keyframe(root)

    output_path = output_root / f"{urdf_path.stem}_controlled.xml"
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path


def _make_mesh_paths_absolute(root: ET.Element, xml_dir: Path) -> None:
    asset = root.find("asset")
    if asset is None:
        return

    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        mesh_path = Path(file_attr)
        if not mesh_path.is_absolute():
            mesh.set("file", str((xml_dir / mesh_path).resolve()))


def _ensure_world_environment(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    if asset.find("texture[@name='skybox']") is None:
        ET.SubElement(asset, "texture", {
            "name": "skybox", "type": "skybox", "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "512",
        })
    if asset.find("texture[@name='checker']") is None:
        ET.SubElement(asset, "texture", {
            "name": "checker", "type": "2d", "builtin": "checker",
            "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3", "width": "512", "height": "512",
            "mark": "cross", "markrgb": ".8 .8 .8",
        })
    if asset.find("material[@name='grid']") is None:
        ET.SubElement(asset, "material", {
            "name": "grid", "texture": "checker", "texrepeat": "1 1",
            "texuniform": "true", "reflectance": "0.2",
        })

    if worldbody.find("light") is None:
        ET.SubElement(worldbody, "light", {
            "pos": "0 0 3", "dir": "0 0 -1", "directional": "true", "castshadow": "false",
        })

    existing_floor = next((g for g in worldbody.findall("geom") if g.get("name") == "floor"), None)
    if existing_floor is None:
        ET.SubElement(worldbody, "geom", {
            "name": "floor", "type": "plane", "size": "0 0 1", "pos": "0 0 0", "material": "grid",
        })
    else:
        existing_floor.set("material", "grid")
        if "rgba" in existing_floor.attrib:
            del existing_floor.attrib["rgba"]

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    if visual.find("headlight") is None:
        ET.SubElement(visual, "headlight", {"ambient": ".1 .1 .1", "diffuse": ".6 .6 .6", "specular": ".3 .3 .3"})
    if visual.find("rgba") is None:
        ET.SubElement(visual, "rgba", {"haze": ".15 .25 .35 1"})
    if visual.find("global") is None:
        ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})
    if visual.find("quality") is None:
        ET.SubElement(visual, "quality", {"shadowsize": "1024"})

    # Use MuJoCo default gravity (0 0 -9.81). Only insert an <option> if absent.
    if root.find("option") is None:
        root.insert(0, ET.Element("option", {"gravity": "0 0 -9.81"}))


# Material appearance per part. The legs and parallel links read as dark glossy
# carbon fibre; the wheels read as matte black rubber, so the robot is no longer
# one uniform grey casting.
_LINK_MATERIALS: dict[str, dict[str, str]] = {
    "carbon_fiber": {
        "rgba": "0.11 0.11 0.12 1",
        "specular": "0.5",
        "shininess": "0.6",
        "reflectance": "0.12",
    },
    "tire_black": {
        "rgba": "0.02 0.02 0.02 1",
        "specular": "0.2",
        "shininess": "0.25",
        "reflectance": "0.0",
    },
}

# Link body name prefix -> material. Legs (link1 upper + link2 lower) and the
# passive parallel link (link3) are carbon fibre; the wheels are matte black.
_LINK_PREFIX_MATERIALS: dict[str, str] = {
    "link1_left": "carbon_fiber",
    "link1_right": "carbon_fiber",
    "link2_left": "carbon_fiber",
    "link2_right": "carbon_fiber",
    "link3_left": "carbon_fiber",
    "link3_right": "carbon_fiber",
    "wheel_left": "tire_black",
    "wheel_right": "tire_black",
}


def _apply_link_materials(root: ET.Element) -> None:
    """Tint leg/link mesh geoms by material so the robot is not uniform grey.

    Each link emits BOTH a "<link>_visual" and a coincident "<link>_collision"
    mesh geom, and both are visible (group 1). Tinting only "_visual" leaves the
    grey "_collision" duplicate rendering on top, so the robot stays uniform
    grey. We therefore tint every mesh geom of the link. MuJoCo also lets a
    geom's rgba override its material colour, so set rgba explicitly for the
    diffuse colour and keep the material for the specular/gloss response.
    """
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for mat_name, attrs in _LINK_MATERIALS.items():
        if asset.find(f"material[@name='{mat_name}']") is None:
            ET.SubElement(asset, "material", {"name": mat_name, **attrs})

    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    for geom in worldbody.iter("geom"):
        if geom.get("type") != "mesh":
            continue
        name = geom.get("name", "")
        for prefix, material in _LINK_PREFIX_MATERIALS.items():
            if name.startswith(prefix):
                geom.set("material", material)
                geom.set("rgba", _LINK_MATERIALS[material]["rgba"])
                break


def _ensure_test_terrain(root: ET.Element, terrain: str | None, terrain_side: str, output_root: Path) -> None:
    if terrain is None:
        return
    if terrain != "single_wheel_trapezoid":
        raise ValueError(f"unsupported terrain: {terrain}")
    _add_single_wheel_trapezoid(root, SingleWheelTrapezoidTerrain(side=terrain_side))
    _add_wavy_road(root, WavyRoadTerrain(), output_root)


def _add_single_wheel_trapezoid(root: ET.Element, config: SingleWheelTrapezoidTerrain) -> None:
    if config.side not in {"left", "right"}:
        raise ValueError(f"unsupported terrain side: {config.side}")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML is missing <worldbody>")
    if worldbody.find("geom[@name='single_wheel_trapezoid_platform']") is not None:
        return

    height = float(config.height)
    ramp_length = float(config.ramp_length)
    platform_length = float(config.platform_length)
    width = float(config.width)
    y_start = float(config.y_start)
    x_pos = float(config.side_x if config.side == "left" else -config.side_x)
    if min(height, ramp_length, platform_length, width) <= 0.0:
        raise ValueError("single-wheel trapezoid terrain dimensions must be positive")

    ramp_span = (ramp_length * ramp_length + height * height) ** 0.5
    angle = atan2(height, ramp_length)
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    # Make each ramp a solid wedge instead of a thin floating slab: thicken the
    # tilted box downward until its underside is buried below the floor across
    # the whole span (thickness * cos >= height). The top face — the climbing
    # surface — stays fixed regardless of thickness, so contact behaviour is
    # unchanged; this just closes the open triangular sides.
    thickness = height / cos_angle + 0.005
    ramp_z = 0.5 * height - 0.5 * thickness * cos_angle
    y_half = 0.5 * width

    _terrain_box(
        worldbody,
        name="single_wheel_trapezoid_ramp_up",
        size=(y_half, 0.5 * ramp_span, 0.5 * thickness),
        pos=(x_pos, y_start + 0.5 * ramp_length + 0.5 * thickness * sin_angle, ramp_z),
        euler=(angle, 0.0, 0.0),
    )
    platform_start = y_start + ramp_length
    _terrain_box(
        worldbody,
        name="single_wheel_trapezoid_platform",
        size=(y_half, 0.5 * platform_length, 0.5 * height),
        pos=(x_pos, platform_start + 0.5 * platform_length, 0.5 * height),
    )
    down_start = platform_start + platform_length
    _terrain_box(
        worldbody,
        name="single_wheel_trapezoid_ramp_down",
        size=(y_half, 0.5 * ramp_span, 0.5 * thickness),
        pos=(x_pos, down_start + 0.5 * ramp_length - 0.5 * thickness * sin_angle, ramp_z),
        euler=(-angle, 0.0, 0.0),
    )


def _terrain_box(
    worldbody: ET.Element,
    *,
    name: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    euler: tuple[float, float, float] | None = None,
) -> None:
    attrs = {
        "name": name,
        "type": "box",
        "size": _format_float_triplet(size),
        "pos": _format_float_triplet(pos),
        "contype": "1",
        "conaffinity": "1",
        "rgba": "0.45 0.42 0.34 1",
    }
    if euler is not None:
        attrs["euler"] = _format_float_triplet(euler)
    ET.SubElement(worldbody, "geom", attrs)


def _add_wavy_road(root: ET.Element, config: WavyRoadTerrain, output_root: Path) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML is missing <worldbody>")
    if worldbody.find("geom[@name='wavy_road']") is not None:
        return

    if min(config.length, config.width, config.amplitude, config.wavelength, config.base_depth) <= 0.0:
        raise ValueError("wavy road terrain dimensions must be positive")
    if config.nrow < 2 or config.ncol < 2:
        raise ValueError("wavy road heightfield resolution must be at least 2x2")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    heightmap_path = output_root / "wavy_road_hfield.png"
    iio.imwrite(heightmap_path, _wavy_road_heightmap(config))

    elevation = 2.0 * config.amplitude  # pixel [0,255] maps linearly to [0, 2A]
    if asset.find("hfield[@name='wavy_road_hfield']") is None:
        ET.SubElement(
            asset,
            "hfield",
            {
                "name": "wavy_road_hfield",
                "file": str(heightmap_path),
                "size": _format_float_quad(
                    (
                        0.5 * config.width,
                        0.5 * config.length,
                        elevation,
                        config.base_depth,
                    )
                ),
            },
        )

    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "wavy_road",
            "type": "hfield",
            "hfield": "wavy_road_hfield",
            "pos": _format_float_triplet((0.0, config.y_start + 0.5 * config.length, 0.0)),
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.0 0.02 0.001",
            "rgba": "0.32 0.31 0.28 1",
        },
    )


def _wavy_road_heightmap(config: WavyRoadTerrain) -> np.ndarray:
    y = np.linspace(-0.5 * config.length, 0.5 * config.length, config.nrow)
    x = np.linspace(-0.5 * config.width, 0.5 * config.width, config.ncol)

    # Base washboard sinusoid in [0, 1]: troughs sit at 0, crests at 1.
    base = 0.5 + 0.5 * np.sin(2.0 * np.pi * y / config.wavelength)

    # Give every crest an independent random height. The cycle index steps at
    # the troughs (where base == 0), so scaling per cycle randomises bump
    # heights while keeping the surface continuous at the cycle boundaries.
    cycle = np.floor(y / config.wavelength + 0.25).astype(int)
    cycle -= int(cycle.min())
    rng = np.random.default_rng(config.seed)
    scales = rng.uniform(config.height_min_scale, 1.0, size=int(cycle.max()) + 1)
    profile = base * scales[cycle]

    heights = np.tile(profile[:, None], (1, config.ncol))

    # Smoothstep envelope so the patch fades into the surrounding floor instead
    # of presenting a vertical step at its boundary.
    fade_y = 0.18 * config.length
    fade_x = 0.10 * config.width
    fy = _smoothstep(np.minimum(
        (y + 0.5 * config.length) / fade_y,
        (0.5 * config.length - y) / fade_y,
    ))
    fx = _smoothstep(np.minimum(
        (x + 0.5 * config.width) / fade_x,
        (0.5 * config.width - x) / fade_x,
    ))
    envelope = np.outer(fy, fx)
    heights = np.clip(heights * envelope, 0.0, 1.0)
    return np.rint(heights * 255.0).astype(np.uint8)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _format_float_triplet(values: tuple[float, float, float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _format_float_quad(values: tuple[float, float, float, float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _ensure_root_freejoint(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML is missing <worldbody>")

    base_link = worldbody.find("body[@name='base_link']")
    if base_link is None:
        raise ValueError("MuJoCo XML is missing base_link body")

    for child in base_link:
        if child.tag in {"freejoint", "joint"} and child.get("name") == "root":
            return

    base_link.insert(0, ET.Element("freejoint", {"name": "root"}))


def _ensure_command_slider_joints(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML is missing <worldbody>")
    if worldbody.find("body[@name='command_slider_body']") is not None:
        return

    body = ET.SubElement(worldbody, "body", {"name": "command_slider_body", "pos": "0 0 -10"})
    ET.SubElement(
        body,
        "inertial",
        {
            "mass": "1e-6",
            "pos": "0 0 0",
            "diaginertia": "1e-9 1e-9 1e-9",
        },
    )
    ET.SubElement(body, "joint", {"name": CMD_SLIDER_NAMES[0], "type": "hinge", "axis": "1 0 0", "damping": "0"})
    ET.SubElement(body, "joint", {"name": CMD_SLIDER_NAMES[1], "type": "hinge", "axis": "0 1 0", "damping": "0"})
    ET.SubElement(body, "joint", {"name": CMD_SLIDER_NAMES[2], "type": "hinge", "axis": "0 0 1", "damping": "0"})
    ET.SubElement(body, "joint", {"name": CMD_SLIDER_NAMES[3], "type": "hinge", "axis": "1 1 0", "damping": "0"})


def _configure_geom_collisions(root: ET.Element) -> None:
    # Set ground to contype=1, conaffinity=1 (default, but ensure it)
    # The ground is usually directly in worldbody
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in worldbody.findall("geom"):
            geom.set("contype", "1")
            geom.set("conaffinity", "1")

    # Set all robot geoms to contype=0, conaffinity=1 so they only collide with ground
    base_link = root.find(".//body[@name='base_link']")
    if base_link is not None:
        for geom in base_link.iter("geom"):
            geom_name = geom.get("name", "")
            is_wheel_visual = geom_name in ("wheel_left_geom", "wheel_right_geom")
            if geom.get("type") == "mesh" or is_wheel_visual:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "1")
            else:
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                geom.set("group", "3")

def _add_collision_proxies(root: ET.Element) -> None:
    """Add simple collision primitives to all robot bodies.

    Mesh geoms are visual-only (contype=0, conaffinity=0).  These proxies
    provide lightweight collision with the ground (conaffinity=1) without
    internal self-collision (contype=0).
    """
    # --- Collision proxy definitions ---
    # Each entry: body_name -> (type, size, pos, quat_or_None, friction)
    # Capsules: size="radius half-length", oriented along local X by default.
    # For MuJoCo capsules the cylinder axis is along Z in the geom frame,
    # so we use quat to rotate as needed.

    _PROXY_DEFS: list[dict[str, str]] = [
        # base_link: box approximation
        {
            "body": "base_link",
            "type": "box",
            "size": "0.10 0.05 0.03",
            "pos": "0 0.03 0.025",
        },
        # link1 left/right: short capsule along link (~67mm span)
        {
            "body": "link1_left",
            "type": "capsule",
            "size": "0.008",
            "fromto": "0 0 0 -0.067 0.015 -0.026",
        },
        {
            "body": "link1_right",
            "type": "capsule",
            "size": "0.008",
            "fromto": "0 0 0 -0.067 -0.015 -0.026",
        },
        # link2 left/right: longer capsule (~100mm span to wheel joint)
        {
            "body": "link2_left",
            "type": "capsule",
            "size": "0.008",
            "fromto": "0 0 0 0.076 0 -0.064",
        },
        {
            "body": "link2_right",
            "type": "capsule",
            "size": "0.008",
            "fromto": "0 0 0 0.076 0 -0.064",
        },
        # link3 left/right: short passive capsule (~40mm)
        {
            "body": "link3_left",
            "type": "capsule",
            "size": "0.006",
            "fromto": "0 0 0 -0.04 0 -0.02",
        },
        {
            "body": "link3_right",
            "type": "capsule",
            "size": "0.006",
            "fromto": "0 0 0 -0.04 0 0.02",
        },
        # wheels: sphere proxy (same as before)
        {
            "body": "wheel_left",
            "type": "sphere",
            "size": str(WHEEL_RADIUS),
            "friction": "1.0 0.02 0.001",
        },
        {
            "body": "wheel_right",
            "type": "sphere",
            "size": str(WHEEL_RADIUS),
            "friction": "1.0 0.02 0.001",
        },
    ]

    for defn in _PROXY_DEFS:
        body_name = defn["body"]
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        proxy_name = f"{body_name}_collision_proxy"
        if body.find(f"geom[@name='{proxy_name}']") is not None:
            continue

        attrs: dict[str, str] = {
            "name": proxy_name,
            "type": defn["type"],
            "size": defn["size"],
            "group": "3",
            "contype": "0",
            "conaffinity": "1",
        }
        if "pos" in defn:
            attrs["pos"] = defn["pos"]
        if "fromto" in defn:
            attrs["fromto"] = defn["fromto"]
        if "friction" in defn:
            attrs["friction"] = defn["friction"]

        # For wheel proxies, place at inertial pos if no explicit pos given
        if "pos" not in defn and "fromto" not in defn:
            inertial = body.find("inertial")
            if inertial is not None and inertial.get("pos"):
                attrs["pos"] = inertial.get("pos")
            else:
                attrs["pos"] = "0 0 0"

        ET.SubElement(body, "geom", attrs)


def _replace_actuators(root: ET.Element) -> None:
    for actuator in root.findall("actuator"):
        root.remove(actuator)

    actuator = ET.Element("actuator")
    for joint_name in MODEL_SEMANTICS.wheel_joints + MODEL_SEMANTICS.leg_motor_joints:
        # Torque limits from real motor specs:
        #   wheels: DM-H6215 rated 1.0 Nm (peak 2.0 Nm)
        #   legs:   DM-J4310-2EC peak 12.5 Nm; controller soft-limits
        #           non-EXTEND phases to rated 3.5 Nm.
        ctrlrange = "-1.0 1.0" if joint_name in MODEL_SEMANTICS.wheel_joints else "-12.5 12.5"
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": f"act_{joint_name}",
                "joint": joint_name,
                "ctrllimited": "true",
                "ctrlrange": ctrlrange,
                "gear": "1",
            },
        )
    # cmd_height range is kept inside the LUT's reachable interval.
    # 2026-05-18: lower bound comes from the viewer-low posture used by
    # the side-specific closed-chain LUT. Upper bound remains 0.142 because the
    # current 2.2 kg tune oscillates above it; restoring 0.148 needs retuning.
    for name, ctrlrange in zip(CMD_SLIDER_NAMES, ("-1.0 1.0", "-3.0 3.0", "0.07844 0.142", "0 1")):
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": name,
                "joint": name,
                "ctrllimited": "true",
                "ctrlrange": ctrlrange,
                "gear": "0",
            },
        )
    root.append(actuator)


def _stiffen_leg_motor_joint_limits(root: ET.Element) -> None:
    """让 motor joint 的 range limit 变成接近硬限位。

    默认 solreflimit (~0.02 1) 让 limit 是一种"软弹簧",冲量大时电机可以
    瞬态越过 limit 几十毫弧度。EXTEND 阶段 7 N·m feedforward + 电机惯性
    把 θ 推过 LUT.theta_max 进入奇异区,正是 URDF upper=0.60 设了也挡不住的
    根因。这里把 solreflimit 调到 0.001 (~20× 更刚),solimplimit 调到接近
    硬约束,让越界量降到几毫弧度内。
    """
    motor_joint_names = set(MODEL_SEMANTICS.leg_motor_joints)
    for joint in root.iter("joint"):
        if joint.get("name") in motor_joint_names:
            joint.set("limited", "true")
            joint.set("solreflimit", "0.001 1")
            joint.set("solimplimit", "0.99 0.999 0.0001 0.5 2")


def _ensure_equality_constraints(root: ET.Element) -> None:
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    if equality.find("connect[@name='link23_left_connect']") is not None:
        return

    # Computed local coordinates for the connection holes (link2 <-> link3)
    # These anchors are in the local coordinate frame of link2_left / link2_right
    ET.SubElement(
        equality,
        "connect",
        {
            "name": "link23_left_connect",
            "body1": "link2_left",
            "body2": "link3_left",
            "anchor": "-0.02865 -0.00846 0.02459",
            "solref": "0.005 1",
            "solimp": "0.95 0.99 0.001 0.5 2",
        },
    )

    ET.SubElement(
        equality,
        "connect",
        {
            "name": "link23_right_connect",
            "body1": "link2_right",
            "body2": "link3_right",
            "anchor": "-0.02869 0.00853 0.02452",
            "solref": "0.005 1",
            "solimp": "0.95 0.99 0.001 0.5 2",
        },
    )


# Constraint-consistent symmetric standing pose. Both legs in the "low elbow"
# branch of the four-bar mechanism (link1 passive ≈ 0.98, base passive ≈ 0.72).
# Pre-computed via equality-constraint settling with gravity disabled and
# symmetric joint initialization. Base z lifts the wheel sphere proxies onto
# the floor (z=0). Wheels and command sliders at 0.
# Indices match the order in sim/state.py:model_addresses().
# CoM offset: 当前模型实测 CoM 在 wheel 中心 +Y 方向约 18 mm，pendulum
# 长度约 0.107 m，平衡 pitch 约 -0.17 rad。请用脚本实测后再使用，不要
# 依赖此注释中的具体数字。
_STANDING_KEYFRAME_QPOS = (
    "0 0 0.175 "                          # base x y z
    "1 0 0 0 "                            # base quat (identity)
    "0.752 0.980 0 "                      # right: motor, link1 passive, wheel
    "0.752 0.980 0 "                      # left: motor, link1 passive, wheel
    "0.720 0.720 "                        # base passive joints (link3 right/left)
    "0 0 0 0"                             # cmd_linear_x, cmd_angular_z, cmd_height, cmd_jump
)


def _ensure_standing_keyframe(root: ET.Element) -> None:
    """Add a standing keyframe satisfying the closed-loop equality constraints."""
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    if keyframe.find("key[@name='stand']") is not None:
        return
    ET.SubElement(keyframe, "key", {"name": "stand", "qpos": _STANDING_KEYFRAME_QPOS})
