import numpy as np
from typing import List
from .types import Candidate
from src.geometry.quadtree import compute_morton_code, p_adic_distance_4

def run_stage2(search_img: np.ndarray, candidates: List[Candidate], config: dict) -> List[Candidate]:
    """
    Stage 2: 4-adic Quadtree Pruning
    Refines shortlist boundaries based on ultrametric branch-and-bound.
    """
    if config.get('exhaustive_fallback', False):
        return candidates
        
    # We simulate the quadtree pruning on the candidate list by calculating
    # Lipschitz bounds around the highest scoring candidates.
    L_mult = config.get('lipschitz_constant_multiplier', 1.5)
    
    # Estimate Lipschitz constant L from gradient magnitude
    gy, gx = np.gradient(search_img)
    L_base = np.max(np.sqrt(gx**2 + gy**2))
    L = L_base * L_mult
    
    best_score = max([c.score for c in candidates]) if candidates else 0.0
    
    survivors = []
    
    for cand in candidates:
        # Check pruning bound using 4-adic proxy
        # Convert coords to Morton codes (integer coords)
        cand_x_int = int(cand.x)
        cand_y_int = int(cand.y)
        
        # Center of search image for max depth reference
        cx = search_img.shape[1] // 2
        cy = search_img.shape[0] // 2
        
        m_cand = compute_morton_code(cand_x_int, cand_y_int, depth=6)
        m_center = compute_morton_code(cx, cy, depth=6)
        
        d4 = p_adic_distance_4(m_cand, m_center)
        
        # Upper bound: Score(v) + L * d4 * cell_size
        upper_bound = cand.score + L * d4 * 10.0 # 10.0 is nominal cell size at this depth
        
        # Branch-and-bound pruning rule
        if upper_bound >= best_score * 0.9: # Give a 10% margin
            survivors.append(cand)
            
    # Typically returns <= 5 candidates
    return survivors[:5]
