import numpy as np
from scipy import sparse

def get_trilinear_weights(rel_pos):
    """Trilinear weights for 8 neighbors."""
    wx = [1 - rel_pos[0], rel_pos[0]]
    wy = [1 - rel_pos[1], rel_pos[1]]
    wz = [1 - rel_pos[2], rel_pos[2]]
    
    weights = []
    offsets = []
    
    for k in range(2):
        for j in range(2):
            for i in range(2):
                w = wx[i] * wy[j] * wz[k]
                weights.append(w)
                offsets.append((i, j, k))
                
    return weights, offsets

def coord_to_idx(coord, res):
    """Convert (x, y, z) to linear index in flattened grid."""
    return coord[0] * res * res + coord[1] * res + coord[2]

def idx_to_coord(idx, res):
    """Convert linear index to (x, y, z)."""
    x = idx // (res * res)
    remainder = idx % (res * res)
    y = remainder // res
    z = remainder % res
    return (x, y, z)

def splat_normals_dense(points, normals, resolution):
    """
    Distributes normal vectors to a DENSE grid.
    Returns: V (res^3, 3) vector field.
    """
    num_voxels = resolution ** 3
    V = np.zeros((num_voxels, 3), dtype=np.float64)
    
    scaled_points = points * (resolution - 1)
    
    for i in range(len(points)):
        p = scaled_points[i]
        n = normals[i]
        
        base = np.floor(p).astype(int)
        base = np.clip(base, 0, resolution - 2)
        rel = p - base
        
        weights, offsets = get_trilinear_weights(rel)
        
        for w, off in zip(weights, offsets):
            neighbor = (base[0] + off[0], base[1] + off[1], base[2] + off[2])
            if all(0 <= neighbor[d] < resolution for d in range(3)):
                idx = coord_to_idx(neighbor, resolution)
                V[idx] += n * w
                
    return V

def compute_divergence_dense(V, resolution):
    """
    Computes divergence of vector field V on a DENSE grid.
    Uses central differences.
    """
    num_voxels = resolution ** 3
    div = np.zeros(num_voxels, dtype=np.float64)
    
    for idx in range(num_voxels):
        coord = idx_to_coord(idx, resolution)
        x, y, z = coord
        
        # dVx/dx
        if x > 0 and x < resolution - 1:
            idx_xm = coord_to_idx((x-1, y, z), resolution)
            idx_xp = coord_to_idx((x+1, y, z), resolution)
            div[idx] += (V[idx_xp, 0] - V[idx_xm, 0]) / 2.0
            
        # dVy/dy
        if y > 0 and y < resolution - 1:
            idx_ym = coord_to_idx((x, y-1, z), resolution)
            idx_yp = coord_to_idx((x, y+1, z), resolution)
            div[idx] += (V[idx_yp, 1] - V[idx_ym, 1]) / 2.0
            
        # dVz/dz
        if z > 0 and z < resolution - 1:
            idx_zm = coord_to_idx((x, y, z-1), resolution)
            idx_zp = coord_to_idx((x, y, z+1), resolution)
            div[idx] += (V[idx_zp, 2] - V[idx_zm, 2]) / 2.0
            
    return div

def build_laplacian_dense(resolution, alpha=1e-5):
    """
    Builds 7-point Laplacian on a DENSE grid.
    Returns sparse matrix A = (L + alpha*I).
    """
    num_voxels = resolution ** 3
    
    rows = []
    cols = []
    data = []
    
    for idx in range(num_voxels):
        coord = idx_to_coord(idx, resolution)
        x, y, z = coord
        
        diag_val = 6.0 + alpha
        
        # Add diagonal
        rows.append(idx)
        cols.append(idx)
        data.append(diag_val)
        
        # Add off-diagonals for each neighbor
        neighbors = [
            (x-1, y, z), (x+1, y, z),
            (x, y-1, z), (x, y+1, z),
            (x, y, z-1), (x, y, z+1)
        ]
        
        for n_coord in neighbors:
            if all(0 <= n_coord[d] < resolution for d in range(3)):
                n_idx = coord_to_idx(n_coord, resolution)
                rows.append(idx)
                cols.append(n_idx)
                data.append(-1.0)
                
    A = sparse.coo_matrix((data, (rows, cols)), shape=(num_voxels, num_voxels))
    return A.tocsr()
