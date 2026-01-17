import numpy as np
import skimage.measure

def extract_isosurface_dense(grid, x, density_threshold=0.0):
    """
    Extracts mesh using Marching Cubes on a densified grid.
    """
    res = grid.resolution
    
    # Limit resolution for densification to avoid OOM
    if res > 512:
        print("Warning: Resolution {res} too high for naive dense MC. Crashing imminent if not careful.")
    
    volume = np.zeros((res, res, res), dtype=np.float32)
    
    # Fill sparse values
    for coord, idx in grid.iter_nodes():
        if idx < len(x):
            volume[coord] = x[idx]
            
    # Theoretical iso-value for Poisson reconstruction is 0.
    level = 0.0
        
    print(f"Running Marching Cubes at level={level:.4f}...")
    verts, faces, normals, values = skimage.measure.marching_cubes(volume, level=level)
    
    # Transform vertices to unit box [0, 1]
    verts = verts / (res - 1)
    
    return verts, faces
