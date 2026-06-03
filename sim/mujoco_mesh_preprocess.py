from __future__ import annotations

import copy
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable


MeshLoader = Callable[[Path], object]
MeshSimplifier = Callable[[object, int], object]


def prepare_mujoco_xml(
    xml_path: Path,
    face_limit: int = 200_000,
    cache_dir: Path | None = None,
    mesh_loader: MeshLoader | None = None,
    simplifier: MeshSimplifier | None = None,
) -> Path:
    del simplifier
    xml_path = xml_path.expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        return xml_path

    output_root = Path(cache_dir) if cache_dir is not None else Path(tempfile.mkdtemp(prefix="mojoco_lqr_mesh_"))
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    changed = False
    for mesh in list(asset.findall("mesh")):
        file_attr = mesh.get("file")
        if not file_attr:
            continue

        source_path = (xml_path.parent / file_attr).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Mesh file not found: {source_path}")

        face_count = _stl_face_count(source_path, mesh_loader)
        if face_count <= face_limit:
            mesh.set("file", str(source_path))
            continue

        mesh_name = mesh.get("name")
        if not mesh_name:
            raise ValueError(f"Oversized mesh is missing a name: {source_path}")

        split_paths = _split_binary_stl(source_path, output_root, face_limit)
        replacement_names: list[str] = []
        asset_index = list(asset).index(mesh)
        asset.remove(mesh)

        for offset, split_path in enumerate(split_paths):
            replacement_name = f"{mesh_name}_part_{offset + 1:03d}"
            replacement_names.append(replacement_name)
            replacement_mesh = ET.Element("mesh", dict(mesh.attrib))
            replacement_mesh.set("name", replacement_name)
            replacement_mesh.set("file", str(split_path))
            asset.insert(asset_index + offset, replacement_mesh)

        _expand_mesh_geoms(root, mesh_name, replacement_names)
        changed = True

    if not changed:
        return xml_path

    rewritten_xml = output_root / xml_path.name
    tree.write(rewritten_xml, encoding="utf-8", xml_declaration=False)
    return rewritten_xml


def _stl_face_count(path: Path, mesh_loader: MeshLoader | None = None) -> int:
    binary_face_count = _binary_stl_face_count(path)
    if binary_face_count is not None:
        return binary_face_count

    if mesh_loader is not None:
        return _face_count(mesh_loader(path))

    raise RuntimeError(
        f"Only binary STL meshes are supported for exact MuJoCo preprocessing: {path}"
    )


def _binary_stl_face_count(path: Path) -> int | None:
    if path.suffix.lower() != ".stl":
        return None

    file_size = path.stat().st_size
    if file_size < 84:
        return None

    with path.open("rb") as stl_file:
        stl_file.seek(80)
        face_count = int.from_bytes(stl_file.read(4), "little")

    expected_size = 84 + face_count * 50
    if expected_size != file_size:
        return None

    return face_count


def _split_binary_stl(source_path: Path, output_root: Path, face_limit: int) -> list[Path]:
    face_count = _binary_stl_face_count(source_path)
    if face_count is None:
        raise RuntimeError(f"Cannot split non-binary STL mesh exactly: {source_path}")

    output_paths: list[Path] = []
    with source_path.open("rb") as source_file:
        header = source_file.read(80)
        source_file.read(4)
        remaining_faces = face_count
        chunk_index = 1

        while remaining_faces > 0:
            chunk_faces = min(face_limit, remaining_faces)
            chunk_path = output_root / f"{source_path.stem}-part-{chunk_index:03d}.stl"
            with chunk_path.open("wb") as chunk_file:
                chunk_file.write(header)
                chunk_file.write(chunk_faces.to_bytes(4, "little"))
                chunk_file.write(source_file.read(chunk_faces * 50))
            output_paths.append(chunk_path)
            remaining_faces -= chunk_faces
            chunk_index += 1

    return output_paths


def _expand_mesh_geoms(root: ET.Element, mesh_name: str, replacement_names: list[str]) -> None:
    for parent in root.iter():
        children = list(parent)
        replaced = False
        expanded_children: list[ET.Element] = []
        for child in children:
            if child.tag == "geom" and child.get("mesh") == mesh_name:
                expanded_children.extend(_clone_mesh_geom(child, replacement_names))
                replaced = True
                continue
            expanded_children.append(child)

        if replaced:
            parent[:] = expanded_children


def _clone_mesh_geom(geom: ET.Element, replacement_names: list[str]) -> list[ET.Element]:
    cloned_geoms: list[ET.Element] = []
    geom_name = geom.get("name")
    for offset, replacement_name in enumerate(replacement_names):
        cloned_geom = copy.deepcopy(geom)
        cloned_geom.set("mesh", replacement_name)
        if geom_name:
            cloned_geom.set("name", f"{geom_name}_part_{offset + 1:03d}")
        cloned_geoms.append(cloned_geom)
    return cloned_geoms


def _face_count(mesh: object) -> int:
    faces = getattr(mesh, "faces", None)
    if faces is None:
        raise TypeError("Loaded mesh does not expose faces")
    return len(faces)
