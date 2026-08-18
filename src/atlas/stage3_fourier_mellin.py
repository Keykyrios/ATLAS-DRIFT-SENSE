"""Stage 3: Fourier-Mellin coarse pose estimation.

Computes closed-form initial (scale, rotation, translation) estimates
for each surviving candidate via log-polar phase correlation (Reddy-Chatterji).
The scale ratio is measured, not assumed, satisfying the cross-magnification
requirement of the problem statement.
"""

import numpy as np
import cv2
from typing import List
from .types import Candidate, PoseEstimate
from src.geometry.fourier_mellin import fourier_mellin_pose

def run_stage3(ref_img: np.ndarray, search_img: np.ndarray, candidates: List[Candidate], config: dict) -> List[PoseEstimate]:
    """
    Stage 3: Fourier-Mellin Coarse Pose
    Computes coarse pose (scale, rotation, tx, ty) for each surviving candidate.
    """
    poses = []
    
    rh, rw = ref_img.shape
    sh, sw = search_img.shape
    
    for cand in candidates:
        # Extract patch from search image around candidate
        # We need a patch roughly the size of the unscaled reference to compare against it
        patch_w = int(rw / cand.scale)
        patch_h = int(rh / cand.scale)
        
        x_min = int(max(0, cand.x - patch_w/2))
        x_max = int(min(sw, cand.x + patch_w/2))
        y_min = int(max(0, cand.y - patch_h/2))
        y_max = int(min(sh, cand.y + patch_h/2))
        
        patch = search_img[y_min:y_max, x_min:x_max]
        
        # Pad to exactly match reference shape
        padded_patch = np.zeros((rh, rw), dtype=np.float32)
        
        # Resize to reference size for phase correlation
        if patch.shape[0] > 0 and patch.shape[1] > 0:
            resized_patch = cv2.resize(patch, (rw, rh))
            padded_patch = resized_patch
            
        s, th, tx, ty = fourier_mellin_pose(ref_img, padded_patch)
        
        # Adjust scale relative to candidate's baseline
        final_s = cand.scale * s
        final_tx = cand.x + (tx * cand.scale)
        final_ty = cand.y + (ty * cand.scale)
        
        poses.append(PoseEstimate(scale=final_s, theta=th, tx=final_tx, ty=final_ty))
        
    return poses
