import numpy as np
from scipy.sparse.linalg import cg
import skimage.measure
from .formulation import splat_normals_dense, compute_divergence_dense, build_laplacian_dense

def solve_poisson_dense(points, normals, resolution, alpha=1e-5):
    """
    Solves Screened Poisson on a DENSE grid.
    """
    print(f"  [1/4] Splatting normals to dense grid ({resolution}^3)...")
    V = splat_normals_dense(points, normals, resolution)
    
    print("  [2/4] Computing divergence...")
    b = compute_divergence_dense(V, resolution)
    
    print("  [3/4] Building Laplacian...")
    A = build_laplacian_dense(resolution, alpha)
    
    print(f"  [4/4] Solving ({A.shape[0]} unknowns)...")
    x, info = cg(A, b, rtol=1e-6, maxiter=1000)
    
    if info != 0:
        print(f"    Warning: CG returned info={info}")
    else:
        print("    Solver converged.")
        
    return x

def extract_isosurface_from_dense(x, resolution):
    """
    Extracts isosurface from the solved dense field.
    """
    # Reshape to 3D volume
    volume = x.reshape((resolution, resolution, resolution))
    
    # Iso-value = mean of field (theoretical=0, but bias can shift it)
    iso_val = np.mean(volume)
    print(f"  Iso-value: {iso_val:.6f}, range: [{volume.min():.4f}, {volume.max():.4f}]")
    
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(volume, level=iso_val)
    except ValueError as e:
        print(f"  Marching Cubes error: {e}")
        return None, None
        
    # Normalize to [0, 1]
    verts = verts / (resolution - 1)
    
    return verts, faces
