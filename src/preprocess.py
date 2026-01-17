import open3d as o3d
import numpy as np

def load_point_cloud(path):
    """Loads a point cloud from a file."""
    print(f"Loading point cloud from {path}...")
    pcd = o3d.io.read_point_cloud(path)
    if not pcd.has_points():
        raise ValueError(f"Could not load point cloud from {path}")
    print(f"Loaded {len(pcd.points)} points.")
    return pcd

def estimate_normals(pcd, k_nn=30):
    """Estimates normals if they are missing."""
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k_nn))
        # Orient the normals to a consistent direction (MST usually good for single view or closed shapes)
        pcd.orient_normals_consistent_tangent_plane(k_nn)
    else:
        print("Point cloud already has normals.")
    return pcd
