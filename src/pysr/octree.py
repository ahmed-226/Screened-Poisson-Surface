import numpy as np

class SparseGrid:
    """
    A simple sparse grid structure to manage active voxels.
    Uses a dictionary to store indices mapping to a linearized state vector.
    """
    def __init__(self, depth):
        self.depth = depth
        self.resolution = 2 ** depth
        # Map (x, y, z) -> index in the global state vector
        self.active_nodes = {} 
        self.reverse_map = [] # index -> (x, y, z)
        
    def add_points(self, points):
        """
        Identify active voxels from a set of normalized points [0, 1].
        
        Args:
            points: (N, 3) numpy array of points in range [0, 1].
        """
        # Scale points to grid coordinates
        grid_coords = (points * self.resolution).astype(np.int32)
        
        # Clip to ensure valid range
        grid_coords = np.clip(grid_coords, 0, self.resolution - 1)
        
        for coord in grid_coords:
            tuple_coord = tuple(coord)
            if tuple_coord not in self.active_nodes:
                self.add_node(tuple_coord)
                
    def add_node(self, coord):
        if coord not in self.active_nodes:
            idx = len(self.active_nodes)
            self.active_nodes[coord] = idx
            self.reverse_map.append(coord)
            
    def expand_buffer(self, steps=1):
        """Expands the set of active nodes by adding neighbors (padding)."""
        directions = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1)
        ]
        
        for _ in range(steps):
            current_nodes = list(self.active_nodes.keys())
            for node in current_nodes:
                x, y, z = node
                # 26-connectivity to ensure spline support coverage
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            neighbor = (x + dx, y + dy, z + dz)
                            # Check boundary
                            if (0 <= neighbor[0] < self.resolution and
                                0 <= neighbor[1] < self.resolution and
                                0 <= neighbor[2] < self.resolution):
                                if neighbor not in self.active_nodes:
                                    self.add_node(neighbor)

    def get_num_nodes(self):
        return len(self.active_nodes)
        
    def iter_nodes(self):
        return self.active_nodes.items()
