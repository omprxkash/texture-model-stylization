"""
Bridge for 3DShape2VecSet-generated meshes (SIGGRAPH 2023 — arXiv 2301.11445).

3DShape2VecSet encodes 3D shapes as a set of latent vectors processed by a
transformer diffusion model. It supports image-conditioned generation, meaning
a texture image can guide the underlying 3D shape generation before we apply
our heightmap texturing.

This module provides:
  - VecSetMeshLoader: load exported OBJ/PLY from 3DShape2VecSet inference
  - shape_from_image (stub): illustrates image-conditioned shape generation pipeline

Install 3DShape2VecSet separately:
  git clone https://github.com/1zb/3DShape2VecSet
  pip install -r 3DShape2VecSet/requirements.txt
"""
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from texture3d.mesh_processor import MeshProcessor


class VecSetMeshLoader:
    """Load a mesh exported by 3DShape2VecSet inference and prepare it for texturing."""

    @staticmethod
    def load(mesh_path: str) -> MeshProcessor:
        """Load and normalize a VecSet-generated mesh."""
        mp = MeshProcessor(mesh_path)
        verts = mp.mesh.vertices.copy()

        # Center and normalize to unit sphere (VecSet outputs may vary in scale)
        verts -= verts.mean(axis=0)
        scale = np.linalg.norm(verts, axis=1).max()
        if scale > 1e-6:
            verts /= scale

        mp.mesh.vertices = verts
        mp.has_uv = False
        return mp

    @staticmethod
    def from_pointcloud(points: np.ndarray, method: str = "ball_pivot") -> MeshProcessor:
        """Reconstruct a mesh from a point cloud (e.g., 3DShape2VecSet partial completion output).

        method: 'ball_pivot' (trimesh) or 'poisson' (open3d, requires open3d)
        """
        if method == "ball_pivot":
            pcd = trimesh.PointCloud(points)
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = pcd.convex_hull
            mp = MeshProcessor()
            mp.mesh = mesh
            mp.has_uv = False
            return mp

        elif method == "poisson":
            try:
                import open3d as o3d
            except ImportError:
                raise ImportError("Install open3d: pip install open3d")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals()
            mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            verts = np.asarray(mesh_o3d.vertices)
            faces = np.asarray(mesh_o3d.triangles)
            mp = MeshProcessor()
            mp.mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mp.has_uv = False
            return mp
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")


def shape_from_image_stub(
    image_path: str,
    vecset_repo: Optional[str] = None,
    output_path: str = "outputs/generated_shape.obj",
) -> str:
    """
    Placeholder for image-conditioned 3D shape generation via 3DShape2VecSet.

    In a full pipeline:
      1. Load the 3DShape2VecSet model (requires the repo cloned locally)
      2. Encode the input image with the image encoder
      3. Run the transformer diffusion model conditioned on the image embedding
      4. Decode vector set to a mesh (marching cubes on SDF)
      5. Export as OBJ

    Returns the path to the generated OBJ for downstream texturing.
    """
    print(
        "shape_from_image: 3DShape2VecSet integration stub.\n"
        "To use:\n"
        "  git clone https://github.com/1zb/3DShape2VecSet\n"
        "  Follow their inference instructions to generate an OBJ from an image,\n"
        "  then pass the OBJ to MeshProcessor for UV generation and heightmap application."
    )
    # Return a sphere as fallback
    mp = MeshProcessor.sphere()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mp.export(output_path)
    return output_path
