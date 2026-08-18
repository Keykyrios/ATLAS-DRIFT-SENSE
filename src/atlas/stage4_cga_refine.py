"""Stage 4: Conformal Geometric Algebra (CGA) joint refinement.

Refines the Fourier-Mellin pose estimates to sub-pixel accuracy by
optimizing all four DOF (s, θ, tx, ty) jointly on the Sim(2) Lie group
manifold via the matrix exponential map.
"""

import numpy as np
from typing import List, Tuple
from .types import PoseEstimate, Candidate
from src.geometry.cga2d import CGA2DSim

def run_stage4(ref_img: np.ndarray, search_img: np.ndarray, poses: List[PoseEstimate], config: dict) -> List[Candidate]:
    """
    Stage 4: CGA Joint Refinement
    Refines (s, theta, tx, ty) via versor-manifold optimization.
    """
    max_iter = config.get('max_iterations', 20)
    lr = config.get('learning_rate', 0.05)
    tol = config.get('convergence_threshold', 1e-4)
    
    cga = CGA2DSim(ref_img, search_img)
    
    refined_candidates = []
    
    for pose in poses:
        s, th, tx, ty, ncc, converged = cga.refine(
            init_s=pose.scale,
            init_theta=pose.theta,
            init_tx=pose.tx,
            init_ty=pose.ty,
            max_iter=max_iter,
            lr=lr,
            tol=tol
        )
        
        refined_candidates.append(Candidate(x=tx, y=ty, scale=s, score=ncc, rotation=th))
        
    # Sort by refined NCC score
    refined_candidates.sort(key=lambda c: c.score, reverse=True)
    return refined_candidates
