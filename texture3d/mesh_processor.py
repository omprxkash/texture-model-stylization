"""
3D mesh loading and UV parameterization using trimesh.

Loads OBJ / STL / PLY meshes (or generates primitives) and produces
a UV-parameterized mesh ready for heightmap displacement.

Design notes:
- If the mesh already has UV coordinates, they are preserved.
- If not, we generate UVs via angle-based unwrapping (trimesh built-in).
- xatlas is used when available for higher-quality parameterization.

Related work: 3DShape2VecSet (SIGGRAPH 2023) uses vector set representations
for 3D shape diffusion; our pipeline applies after shape generation, treating
the output mesh as input here.
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh


class MeshProcessor:
    """Load, process, and UV-parameterize 3D meshes."""

    def __init__(self, mesh_path: Optional[str] = None):
        self.mesh: Optional[trimesh.Trimesh] = None
        self.has_uv: bool = False
        if mesh_path:
            self.load(mesh_path)

    def load(self, path: str) -> "MeshProcessor":
        scene_or_mesh = trimesh.load(path, force="mesh")
        if isinstance(scene_or_mesh, trimesh.Scene):
            geoms = list(scene_or_mesh.geometry.values())
            self.mesh = trimesh.util.concatenate(geoms)
        else:
            self.mesh = scene_or_mesh

        if hasattr(self.mesh, "visual") and hasattr(self.mesh.visual, "uv"):
            uv = self.mesh.visual.uv
            self.has_uv = uv is not None and len(uv) > 0
        print(f"Loaded: {path} | vertices={len(self.mesh.vertices)} faces={len(self.mesh.faces)} UV={self.has_uv}")
        return self

    @classmethod
    def sphere(cls, radius: float = 1.0, subdivisions: int = 3) -> "MeshProcessor":
        mp = cls()
        mp.mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        mp.has_uv = False
        print(f"Created sphere: vertices={len(mp.mesh.vertices)}")
        return mp

    @classmethod
    def plane(cls, size: float = 1.0, subdivisions: int = 32) -> "MeshProcessor":
        mp = cls()
        mp.mesh = trimesh.creation.box(extents=(size, size, 0.01))
        mp.has_uv = False
        return mp

    @classmethod
    def box(cls, extents: Tuple[float, float, float] = (1, 1, 1)) -> "MeshProcessor":
        mp = cls()
        mp.mesh = trimesh.creation.box(extents=extents)
        mp.has_uv = False
        return mp

    def generate_uv(self, method: str = "auto") -> "MeshProcessor":
        """Assign UV coordinates to the mesh.

        method:
          'auto'   — use xatlas if installed, else angle-based
          'xatlas' — xatlas parameterization (best quality, pip install xatlas)
          'sphere' — spherical projection (fast, good for convex shapes)
          'angle'  — trimesh angle-based unwrapping
        """
        if self.has_uv:
            print("Mesh already has UV coordinates.")
            return self

        if method in ("auto", "xatlas"):
            try:
                return self._uv_xatlas()
            except (ImportError, Exception) as e:
                if method == "xatlas":
                    raise
                print(f"xatlas not available ({e}), falling back to spherical UV")
                method = "sphere"

        if method == "sphere":
            return self._uv_spherical()
        elif method == "angle":
            return self._uv_angle()
        else:
            raise ValueError(f"Unknown UV method: {method}")

    def _uv_xatlas(self) -> "MeshProcessor":
        import xatlas
        vmapping, indices, uvs = xatlas.parametrize(self.mesh.vertices, self.mesh.faces)
        new_verts = self.mesh.vertices[vmapping]
        self.mesh = trimesh.Trimesh(vertices=new_verts, faces=indices, process=False)
        self.mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        self.has_uv = True
        print(f"UV generated via xatlas: {len(uvs)} UV coords")
        return self

    def _uv_spherical(self) -> "MeshProcessor":
        verts = self.mesh.vertices.copy()
        centroid = verts.mean(axis=0)
        verts -= centroid
        norms = np.linalg.norm(verts, axis=1, keepdims=True) + 1e-8
        verts_n = verts / norms

        u = 0.5 + np.arctan2(verts_n[:, 0], verts_n[:, 2]) / (2 * np.pi)
        v = 0.5 - np.arcsin(np.clip(verts_n[:, 1], -1, 1)) / np.pi
        uvs = np.stack([u, v], axis=1)

        self.mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
        self.has_uv = True
        print(f"UV generated via spherical projection: {len(uvs)} UV coords")
        return self

    def _uv_angle(self) -> "MeshProcessor":
        # Angle-based unwrapping via trimesh's built-in method
        try:
            unwrapped = trimesh.unwrap(self.mesh)
            self.mesh = unwrapped
            self.has_uv = True
            print("UV generated via angle-based unwrapping")
        except Exception as e:
            print(f"Angle-based unwrapping failed: {e}. Falling back to spherical.")
            return self._uv_spherical()
        return self

    def get_uv(self) -> np.ndarray:
        if not self.has_uv:
            self.generate_uv()
        return np.array(self.mesh.visual.uv)

    def export(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.mesh.export(path)
        print(f"Exported: {path}")

    @property
    def vertices(self) -> np.ndarray:
        return self.mesh.vertices

    @property
    def faces(self) -> np.ndarray:
        return self.mesh.faces

    @property
    def vertex_normals(self) -> np.ndarray:
        return self.mesh.vertex_normals
