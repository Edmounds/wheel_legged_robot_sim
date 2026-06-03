"""Feature-preserving decimation of base_link.stl via PyMeshLab.

Uses Quadric Edge Collapse with planar-quadric weighting and optimal placement,
so flat regions decimate aggressively while curved regions (holes, cylinders,
fillets) keep their roundness. Always reads from base_link.stl.orig.

Usage:
    uv run python scripts/decimate_base_link.py [target_faces] [--quality F] [--no-planar]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pymeshlab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_faces", type=int, nargs="?", default=30000)
    parser.add_argument("--quality", type=float, default=1.0,
                        help="Quality threshold 0..1; higher rejects bad triangles (default 1.0)")
    parser.add_argument("--no-planar", action="store_true",
                        help="Disable planar-quadric weighting (worse for boxy shapes)")
    args = parser.parse_args()

    dst = Path("sim/robot/meshes/base_link.stl")
    src = dst.with_suffix(".stl.orig")
    if not src.is_file():
        raise FileNotFoundError(f"Missing pristine backup: {src}")

    ms = pymeshlab.MeshSet()
    # PyMeshLab dispatches by extension, so feed it a .stl symlink to .orig.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_stl = Path(td) / "base_link.stl"
        tmp_stl.write_bytes(src.read_bytes())
        ms.load_new_mesh(str(tmp_stl))
    in_faces = ms.current_mesh().face_number()
    in_verts = ms.current_mesh().vertex_number()
    print(f"input  (from .orig):  verts={in_verts:,}  faces={in_faces:,}")

    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=int(args.target_faces),
        qualitythr=float(args.quality),
        preserveboundary=True,
        boundaryweight=2.0,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=not args.no_planar,
        planarweight=0.002,
        qualityweight=False,
        autoclean=True,
        selected=False,
    )

    out_faces = ms.current_mesh().face_number()
    out_verts = ms.current_mesh().vertex_number()
    print(f"output:               verts={out_verts:,}  faces={out_faces:,}")

    ms.save_current_mesh(str(dst), binary=True, save_face_color=False)
    print(f"wrote: {dst} ({dst.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()