import cv2
import numpy as np
from typing import Tuple

def crop_reference(
    search_img: np.ndarray, 
    scale_ratio: float, 
    rotation_deg: float,
    center_x: float,
    center_y: float,
    ref_size: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a large true structure (search_img), applies the true transformation:
    1. Extracts a smaller view corresponding to the reference field of view.
    2. Applies scaling (10x zoom = 1/0.1 shrink factor).
    3. Applies rotation.
    
    Returns:
        (reference_img, true_transform_matrix_for_validation)
    """
    # The reference is a 1000x1000 image that physically corresponds to a small patch in the search image.
    # The scale_ratio is ~0.1, meaning the patch in the search image is ~100x100.
    # We want to map points in the Reference image (0 to 1000) back to the Search image.
    # So M maps Ref -> Search.
    # M must apply scale_ratio (e.g. 0.1) and rotation, and map the center of Ref (500,500) to (center_x, center_y).
    
    # Affine matrix mapping Ref to Search:
    # 1. Translate Ref center (ref_size/2, ref_size/2) to origin.
    # 2. Scale by scale_ratio and rotate.
    # 3. Translate to (center_x, center_y) in Search.
    
    th_rad = np.deg2rad(rotation_deg)
    s = scale_ratio
    
    # R_S maps from origin-centered Ref to origin-centered Search
    R_S = np.array([
        [s * np.cos(th_rad), -s * np.sin(th_rad)],
        [s * np.sin(th_rad), s * np.cos(th_rad)]
    ])
    
    M = np.zeros((2, 3))
    M[:, :2] = R_S
    
    # Center of reference
    cx_ref, cy_ref = ref_size / 2.0, ref_size / 2.0
    
    # M * [cx_ref, cy_ref]^T + t = [center_x, center_y]^T
    # t = [center_x, center_y]^T - M * [cx_ref, cy_ref]^T
    t = np.array([center_x, center_y]) - R_S @ np.array([cx_ref, cy_ref])
    M[:, 2] = t
    
    # We want to produce the Reference image. We need to warp the Search image.
    # cv2.warpAffine with WARP_INVERSE_MAP uses M to map destination (Ref) pixels back to source (Search).
    # Since our M maps Ref -> Search, this is exactly what we need!
    
    ref_img = cv2.warpAffine(
        search_img, M, (ref_size, ref_size), 
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return ref_img, M
