"""
Voxelizer: Converts continuous 3D point clouds into discrete 3D occupancy voxel grids (32x32x32)
and reconstructs meshes via Marching Cubes.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import scipy.ndimage


def normalize_point_cloud(
    pc: np.ndarray,
    scale_factor: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Centers the point cloud at the origin and normalizes it to fit strictly inside [-0.5, 0.5]^3.
    
    Args:
        pc: (N, 3) point cloud
        scale_factor: scaling cushion (default 0.95 to keep points within bounds)
    Returns:
        pc_normalized: (N, 3) normalized points
        centroid: (3,) original center
        max_dist: original maximum bounding radius (in physical units, e.g. mm)
    """
    centroid = np.mean(pc, axis=0)
    shifted = pc - centroid
    max_dist = np.max(np.linalg.norm(shifted, axis=1))
    if max_dist < 1e-6:
        max_dist = 1.0
    pc_normalized = (shifted / (2.0 * max_dist)) * scale_factor
    return pc_normalized.astype(np.float32), centroid.astype(np.float32), float(max_dist)


def points_to_voxels(
    pc: np.ndarray,
    voxel_res: int = 32,
    dilate: bool = True
) -> np.ndarray:
    """
    Converts a normalized (N, 3) point cloud in [-0.5, 0.5]^3 to a (voxel_res, voxel_res, voxel_res)
    binary occupancy grid.
    
    Args:
        pc: (N, 3) numpy array in range approx [-0.5, 0.5]
        voxel_res: resolution of the grid (default 32)
        dilate: whether to apply a 1-voxel dilation to connect thin surfaces (e.g. chair legs)
    Returns:
        voxels: (voxel_res, voxel_res, voxel_res) uint8 or float32 binary array
    """
    grid = np.zeros((voxel_res, voxel_res, voxel_res), dtype=np.float32)
    
    # Map [-0.5, 0.5] to [0, voxel_res - 1]
    coords = np.floor((pc + 0.5) * (voxel_res - 1) + 0.5).astype(np.int32)
    coords = np.clip(coords, 0, voxel_res - 1)
    
    # Set occupied cells
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    
    if dilate:
        struct = scipy.ndimage.generate_binary_structure(3, 1)  # 6-connectivity
        grid = scipy.ndimage.binary_dilation(grid, structure=struct).astype(np.float32)
        
    return grid


def voxels_to_mesh(
    voxels: np.ndarray,
    threshold: float = 0.5
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts surface mesh from a voxel grid using Marching Cubes.
    
    Args:
        voxels: (V, V, V) 3D numpy array
        threshold: occupancy cutoff (default 0.5)
    Returns:
        verts: (M, 3) vertices in [-0.5, 0.5] range
        faces: (F, 3) triangle indices
    """
    import skimage.measure
    
    # Pad grid with 0 to ensure watertight closed surfaces at boundary
    padded = np.pad(voxels, 1, mode="constant", constant_values=0)
    res = voxels.shape[0]
    
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(padded, level=threshold)
        # Shift back from padding and normalize to [-0.5, 0.5]
        verts = (verts - 1.0) / (res - 1.0) - 0.5
        return verts.astype(np.float32), faces.astype(np.int32)
    except (ValueError, RuntimeError):
        # Empty or degenerate grid
        return None
