"""Stage 1: Hyperdimensional Computing (HDC) multi-scale candidate shortlist.

Reduces the O(N^2) candidate space to a small working set (K ≤ 20) by
encoding patches into high-dimensional bipolar vectors and ranking by
Hamming-based cosine similarity across scale and rotation hypotheses.
"""

import numpy as np
from typing import List
from src.hdc.hypervectors import BipolarHypervectorSpace
from src.hdc.shortlist import compute_hdc_shortlist
from .types import Candidate

def run_stage1(ref_img: np.ndarray, search_img: np.ndarray, config: dict) -> List[Candidate]:
    """
    Stage 1: HDC Shortlist
    Reduces candidate space to a small working set.
    """
    d_dim = config.get('d_dimensions', 2048)
    top_k = config.get('top_k', 20)
    stride = config.get('stride', 15)
    scale_hyps = config.get('scale_hypotheses', [0.095, 0.1, 0.105])
    rot_hyps = config.get('rotation_hypotheses', [-2.0, 0.0, 2.0])
    
    hv_space = BipolarHypervectorSpace(d=d_dim)
    
    raw_candidates = compute_hdc_shortlist(
        ref_img, search_img, hv_space, 
        scale_hypotheses=scale_hyps,
        rotation_hypotheses=rot_hyps,
        stride=stride,
        top_k=top_k
    )
    
    candidates = []
    for (x, y, s, sim) in raw_candidates:
        candidates.append(Candidate(x=x, y=y, scale=s, score=sim))
        
    return candidates
