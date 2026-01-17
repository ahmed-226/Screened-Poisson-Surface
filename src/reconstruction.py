import open3d as o3d
import numpy as np
from .pysr.solver import solve_poisson_dense, extract_isosurface_from_dense

def run_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False):
    """
    Runs Open3D's Screened Poisson Surface Reconstruction.
    """
    print(f"Running Poisson reconstruction (depth={depth}, scale={scale})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )
    print(f"Reconstruction complete. Generated {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    return mesh, densities

def run_poisson_manual(pcd, depth=6, scale=1.1, alpha=1e-5):
    """
    Runs Manual Screened Poisson Surface Reconstruction using DENSE grid.
    NOTE: depth > 6 can be slow/memory intensive. depth=6 -> 64^3 = 262k voxels.
    """
    print(f"Running MANUAL Poisson reconstruction (depth={depth}, scale={scale})...")
    
    resolution = 2 ** depth
    if resolution > 128:
        print(f"  WARNING: Resolution {resolution}^3 may be slow. Consider depth <= 7.")
    
    # 1. Normalize Points to [0.05, 0.95]
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    extent = (bbox_max - bbox_min).max()
    
    max_dim = extent * scale
    normalized_points = (points - center) / max_dim + 0.5
    normalized_points = np.clip(normalized_points, 0.05, 0.95)
        
    # 2. Solve on dense grid
    x = solve_poisson_dense(normalized_points, normals, resolution, alpha)
    
    # 3. Extract Isosurface
    verts, faces = extract_isosurface_from_dense(x, resolution)
    
    if verts is None or len(verts) == 0:
        print("  Extraction failed.")
        return o3d.geometry.TriangleMesh(), []
        
    # 4. Denormalize Vertices
    verts_world = (verts - 0.5) * max_dim + center
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    
    # Flip triangle winding order to fix outward-facing normals
    # Marching Cubes returns faces with inward-pointing normals for our formulation
    faces_flipped = faces[:, ::-1].astype(np.int32)
    mesh.triangles = o3d.utility.Vector3iVector(faces_flipped)
    
    # Compute vertex normals
    mesh.compute_vertex_normals()
    
    print(f"Manual Reconstruction complete. Generated {len(mesh.vertices)} vertices.")
    return mesh, []
