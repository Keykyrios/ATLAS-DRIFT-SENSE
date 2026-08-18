import cv2
import numpy as np
from typing import List, Tuple
from .hypervectors import BipolarHypervectorSpace

def compute_hdc_shortlist(
    ref_img: np.ndarray, 
    search_img: np.ndarray, 
    hv_space: BipolarHypervectorSpace,
    scale_hypotheses: List[float],
    rotation_hypotheses: List[float],
    stride: int = 4,
    top_k: int = 20
) -> List[Tuple[float, float, float, float]]:
    """
    Computes a hyperdimensional multi-scale shortlist in near-linear time
    via a sliding-window array broadcasting approach.
    """
    rh, rw = ref_img.shape
    sh, sw = search_img.shape
    
    # 1. Bundle rotated references (N, D)
    rot_hvs = []
    for theta in rotation_hypotheses:
        M = cv2.getRotationMatrix2D((rw/2, rh/2), theta, 1.0)
        rot_ref = cv2.warpAffine(ref_img, M, (rw, rh))
        rot_hvs.append(hv_space.encode_patch(rot_ref))
        
    bundled_ref_hv = hv_space.bundle(rot_hvs) # (D,)
    
    all_candidates = []
    
    # We must iterate scales, but vectorization over sliding windows eliminates x/y loops
    for s in scale_hypotheses:
        # Instead of extracting patches from search and scaling them down individually,
        # we scale the entire search image once.
        scaled_sh = int(sh * s)
        scaled_sw = int(sw * s)
        
        if scaled_sh < rh or scaled_sw < rw:
            continue
            
        scaled_search = cv2.resize(search_img, (scaled_sw, scaled_sh), interpolation=cv2.INTER_AREA)
        
        # Use stride tricks to extract all patches at once (zero-copy view)
        view = np.lib.stride_tricks.sliding_window_view(scaled_search, (rh, rw))
        
        # view shape is (H_out, W_out, rh, rw). Apply stride
        view_strided = view[::stride, ::stride, :, :]
        H_out, W_out, _, _ = view_strided.shape
        
        # Flatten spatial dimensions for batch processing: (N, rh, rw)
        patches_batch = view_strided.reshape(-1, rh, rw)
        
        # Skip if empty
        if patches_batch.shape[0] == 0:
            continue
            
        # Batch encode
        batch_hvs = hv_space.encode_batch(patches_batch) # (N, D)
        
        # Batch cosine similarity
        sims = hv_space.cosine_similarity_batch(bundled_ref_hv, batch_hvs) # (N,)
        
        # Remap N back to (y_idx, x_idx) in the strided view
        y_indices, x_indices = np.unravel_index(np.arange(len(sims)), (H_out, W_out))
        
        # Map back to original search image coordinates
        orig_y = (y_indices * stride) / s
        orig_x = (x_indices * stride) / s
        
        # Center of the patch in original coordinates
        center_x = orig_x + (rw / s) / 2.0
        center_y = orig_y + (rh / s) / 2.0
        
        # Create candidate tuples
        for i in range(len(sims)):
            all_candidates.append((center_x[i], center_y[i], s, sims[i]))
            
    # Sort all hypotheses across all scales
    all_candidates.sort(key=lambda c: c[3], reverse=True)
    return all_candidates[:top_k]
