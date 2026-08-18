import numpy as np
import cv2
from typing import List
from .types import Candidate
from src.topology.persistence import compute_persistence_diagram, bottleneck_distance

def run_stage5(ref_img: np.ndarray, search_img: np.ndarray, refined_candidates: List[Candidate], config: dict) -> List[Candidate]:
    """
    Stage 5: Persistent Homology Tie-break
    Resolves ambiguity among near-tied candidates using bottleneck distance on D1 diagrams.
    """
    if not refined_candidates:
        return []
        
    delta = config.get('ncc_noise_floor_delta', 0.015)
    max_ncc = refined_candidates[0].score
    
    # Find tied set
    tied_set = [c for c in refined_candidates if c.score >= max_ncc - delta]
    
    if len(tied_set) == 1:
        return tied_set
        
    # Compute reference diagram
    ref_d1 = compute_persistence_diagram(ref_img)
    
    rh, rw = ref_img.shape
    
    # Score tied candidates by topological distance
    topo_scores = []
    for cand in tied_set:
        # Extract properly scaled/rotated patch from search image
        M = cv2.getRotationMatrix2D((cand.x, cand.y), cand.rotation, cand.scale)
        M[0, 2] += (rw / 2) - cand.x
        M[1, 2] += (rh / 2) - cand.y
        
        cand_patch = cv2.warpAffine(search_img, M, (rw, rh), flags=cv2.INTER_LINEAR)
        
        cand_d1 = compute_persistence_diagram(cand_patch)
        
        d_topo = bottleneck_distance(ref_d1, cand_d1)
        topo_scores.append(d_topo)
        
    # Normalize topo scores between 0 and 1 for confidence fusing later
    max_d = max(topo_scores) if max(topo_scores) > 0 else 1.0
    
    # Store topo distance in the Candidate object
    for i, cand in enumerate(tied_set):
        cand.d_topo = topo_scores[i]
        cand.d_topo_norm = topo_scores[i] / max_d
        
    # Sort by topological distance (smaller is better)
    tied_set.sort(key=lambda c: c.d_topo)
    
    # If there is STILL a tie (identical topo distance), it will fall back to center-distance in Stage 6
    return tied_set
