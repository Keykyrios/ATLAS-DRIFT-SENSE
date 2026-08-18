import numpy as np
import gudhi

def compute_persistence_diagram(image: np.ndarray) -> list:
    """
    Computes D1 (loop) persistence diagram for a 2D image 
    using Cubical sublevel-set filtration.
    """
    # gudhi CubicalComplex expects 1D array flattened
    flattened = image.flatten()
    cc = gudhi.CubicalComplex(dimensions=image.shape, top_dimensional_cells=flattened)
    cc.compute_persistence()
    
    # Extract D1 (dimension 1 = loops)
    d1 = cc.persistence_intervals_in_dimension(1)
    
    # If no loops found, return empty list (not None for bottleneck distance)
    if d1 is None or len(d1) == 0:
        return np.array([[0.0, 0.0]])
        
    # Replace infinity with a large finite number for bottleneck computation
    max_val = np.max(image) + 1.0
    d1_clean = []
    for b, d in d1:
        if d == float('inf'):
            d = max_val
        d1_clean.append([b, d])
        
    return np.array(d1_clean)

def bottleneck_distance(diag1: np.ndarray, diag2: np.ndarray) -> float:
    """
    Computes the bottleneck distance between two persistence diagrams.
    """
    return gudhi.bottleneck_distance(diag1, diag2)
