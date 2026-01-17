import numpy as np

def b_spline_generic(t, degree):
    """
    Evaluates B-Spline of given degree at t.
    Currently implements Degree 2 (Quadratic) which is standard for Poisson Reconstruction.
    """
    if degree == 1:
        # Linear B-Spline (Hat function)
        # Support [-1, 1] centered at 0.5?
        # Standard def: centered at 0. [-1, 1]
        # |t| < 1 ? 1 - |t| : 0
        pass
    
    if degree == 2:
        # Quadratic B-Spline
        # Support [-1.5, 1.5]. 
        # But we usually map this to discrete stencil.
        t_abs = abs(t)
        if t_abs <= 0.5:
             return 0.75 - t_abs**2
        elif t_abs <= 1.5:
             return 0.5 * (1.5 - t_abs)**2
        else:
             return 0.0
             
    raise NotImplementedError("Only degree 2 implemented for simplicity")

def b_spline_derivative_generic(t, degree):
    if degree == 2:
        # Derivative of Quadratic B-Spline
        # If |t| <= 0.5: -2t
        # If 0.5 < t <= 1.5: -(1.5 - t) = t - 1.5
        # If -1.5 <= t < -0.5: (1.5 + t)
        if abs(t) <= 0.5:
            return -2 * t
        elif 0.5 < t <= 1.5:
            return t - 1.5
        elif -1.5 <= t < -0.5:
            return t + 1.5
        else:
            return 0.0
    return 0.0

# Precomputed stencils for grid discrete convolution
# For checking value at node i from a function centered at node j, 
# The value depends on dist = |i - j|.
# A quadratic B-spline spans 3 nodes effectively (radius 1.5).
# Stencil offsets: -1, 0, 1
STENCIL_VALS_DEG2 = [0.125, 0.75, 0.125] # B(0)=0.75, B(1)=0.125
STENCIL_DERIV_DEG2 = [0.5, 0.0, -0.5]    # Approximated Central Diff? 
# Actually, precise derivative of B-spline at integer offsets?
# B'(0) = 0.
# B'(1) = -0.5 (from formula t-1.5 at t=1 -> -0.5)
# B'(-1) = 0.5 (from formula t+1.5 at t=-1 -> 0.5)
