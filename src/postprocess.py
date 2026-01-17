import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def clean_mesh(mesh):
    """Removes degenerate triangles, duplicated vertices, etc."""
    print("Cleaning mesh...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    print(f"Cleaned mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    return mesh

def filter_by_density(mesh, densities, quantile=0.01):
    """
    Removes vertices with low density (extrapolated regions).
    
    Args:
        mesh: The reconstructed mesh.
        densities: The density values returned by Poisson reconstruction.
        quantile: The lower quantile of density to prune.
    """
    print(f"Filtering low density regions (quantile={quantile})...")
    densities = np.asarray(densities)
    if len(densities) == 0:
        print("No densities provided, skipping density filter.")
        return mesh

    density_threshold = np.quantile(densities, quantile)
    vertices_to_remove = densities < density_threshold
    
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"Filtered mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    return mesh
