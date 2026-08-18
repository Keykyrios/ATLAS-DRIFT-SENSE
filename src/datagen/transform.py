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
    # The reference is physically a zoomed-in version. 
    # To simulate this, we extract a patch from `search_img` of size `ref_size / scale_ratio`,
    # then scale it down to `ref_size`, and rotate it.
    
    src_h, src_w = search_img.shape
    
    # Base patch size in search coordinates
    patch_size_in_search = int(ref_size / scale_ratio)
    
    # Define affine transform to extract the rotated, scaled patch from the given center
    M = cv2.getRotationMatrix2D((center_x, center_y), rotation_deg, scale_ratio)
    
    # We want the output image to be exactly `ref_size x ref_size`, centered around (center_x, center_y) in the src.
    # We adjust the translation part of M so the requested center ends up in the middle of the ref_size window.
    M[0, 2] += (ref_size / 2) - center_x
    M[1, 2] += (ref_size / 2) - center_y
    
    ref_img = cv2.warpAffine(
        search_img, M, (ref_size, ref_size), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return ref_img, M
