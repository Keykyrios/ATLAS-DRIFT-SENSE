import cv2
import numpy as np
from typing import Tuple

def fourier_mellin_pose(ref_img: np.ndarray, search_patch: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes coarse pose (scale, rotation, tx, ty) via log-polar phase correlation.
    Assumes ref_img and search_patch are same shape (padded/cropped).
    Returns (scale, theta_deg, tx, ty).
    """
    # 1. Hanning window to reduce edge artifacts
    hanning = cv2.createHanningWindow(ref_img.shape[::-1], cv2.CV_64F)
    ref_win = ref_img * hanning
    search_win = search_patch * hanning
    
    # 2. Magnitude spectra
    ref_fft = np.fft.fftshift(np.fft.fft2(ref_win))
    search_fft = np.fft.fftshift(np.fft.fft2(search_win))
    
    ref_mag = np.abs(ref_fft)
    search_mag = np.abs(search_fft)
    
    # High-pass filter to emphasize edges
    h, w = ref_mag.shape
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    hp_filter = 1.0 - np.exp(-(x**2 + y**2) / (2 * (0.05 * min(h, w))**2))
    ref_mag *= hp_filter
    search_mag *= hp_filter
    
    # 3. Log-polar transform
    center = (w / 2, h / 2)
    max_radius = np.sqrt(center[0]**2 + center[1]**2)
    
    ref_logpolar = cv2.warpPolar(ref_mag, (w, h), center, max_radius, cv2.WARP_POLAR_LOG)
    search_logpolar = cv2.warpPolar(search_mag, (w, h), center, max_radius, cv2.WARP_POLAR_LOG)
    
    # 4. Phase correlation for scale and rotation
    res, _ = cv2.phaseCorrelate(ref_logpolar.astype(np.float64), search_logpolar.astype(np.float64))
    
    # Interpret results
    log_base = np.exp(np.log(max_radius) / w)
    scale = log_base ** res[0]
    
    theta_deg = res[1] * (360.0 / h)
    
    # Restrict to expected domains for Drift-Sense (scale ~1.0 since it's already a patch, rotation small)
    if scale > 2.0 or scale < 0.5:
        scale = 1.0
    if theta_deg > 180:
        theta_deg -= 360
        
    # 5. Rotate and scale back for translation
    M = cv2.getRotationMatrix2D(center, theta_deg, scale)
    search_corrected = cv2.warpAffine(search_patch, M, (w, h))
    
    # Phase correlation for translation
    t_res, _ = cv2.phaseCorrelate(ref_img.astype(np.float64), search_corrected.astype(np.float64))
    
    return scale, theta_deg, t_res[0], t_res[1]
