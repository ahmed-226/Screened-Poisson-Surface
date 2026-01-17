import open3d as o3d
import numpy as np

def generate_sphere_point_cloud(radius=1.0, num_points=2000, noise_std=0.0):
    """Generates a synthetic point cloud of a sphere."""
    print(f"Generating sphere (r={radius}, N={num_points}, noise={noise_std})...")
    
    # Random points on sphere
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2*np.pi, num_points)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    points = np.stack((x, y, z), axis=1)
    
    # Add noise
    if noise_std > 0:
        points += np.random.normal(0, noise_std, points.shape)
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Normals (for a sphere at origin, normal is just the normalized position vector)
    # If we want to simulate "estimated" normals, we might not set them here and let preprocess do it.
    # But for a clear ground truth, let's set them.
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd

def visualize(geometries, window_name="Open3D"):
    """Visualizes a list of geometries."""
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
