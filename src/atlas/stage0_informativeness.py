"""Stage 0: Kolmogorov/NCD Informativeness Gate.

Estimates the intrinsic ambiguity of the reference patch before search begins
by computing the Normalized Compression Distance (NCD) between the patch and
its own copy shifted by the dominant lattice period. A low NCD indicates a
highly periodic (and therefore ambiguous) reference, triggering an a-priori
low-confidence flag.

References:
    Li et al., "The Similarity Metric", IEEE Trans. Info. Theory, 2004.
    Cilibrasi & Vitányi, "Clustering by Compression", IEEE Trans. Info. Theory, 2005.
"""

import zlib
import numpy as np
import scipy.signal
from .types import InformativenessReport

def compute_ncd(x: bytes, y: bytes) -> float:
    c_x = len(zlib.compress(x, level=9))
    c_y = len(zlib.compress(y, level=9))
    c_xy = len(zlib.compress(x + y, level=9))
    
    return (c_xy - min(c_x, c_y)) / max(c_x, c_y)

def run_stage0(ref_img: np.ndarray, config: dict) -> InformativenessReport:
    """
    Stage 0: Informativeness Gate
    Estimates dominant lattice period and computes NCD self-shift score.
    """
    tau_max = config.get('tau_max', 50)
    threshold = config.get('ncd_threshold', 0.15)
    
    # Ensure zero mean for autocorrelation
    ref_mean = np.mean(ref_img)
    ref_zero_mean = ref_img - ref_mean
    
    # 2D Autocorrelation via FFT
    autocorr = scipy.signal.fftconvolve(ref_zero_mean, ref_zero_mean[::-1, ::-1], mode='same')
    
    # Find dominant period (ignoring center peak)
    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    
    # Mask center
    mask_radius = 5
    autocorr[cy-mask_radius:cy+mask_radius, cx-mask_radius:cx+mask_radius] = 0
    
    # Find max within tau_max
    y_min = max(0, cy - tau_max)
    y_max = min(h, cy + tau_max)
    x_min = max(0, cx - tau_max)
    x_max = min(w, cx + tau_max)
    
    local_autocorr = autocorr[y_min:y_max, x_min:x_max]
    
    # If image is totally flat
    if np.max(local_autocorr) == 0:
        return InformativenessReport(0.0, 0, 0, True)
        
    idx = np.unravel_index(np.argmax(local_autocorr), local_autocorr.shape)
    tau_y = (idx[0] + y_min) - cy
    tau_x = (idx[1] + x_min) - cx
    
    # Shift reference by tau
    shifted = np.roll(ref_img, shift=(tau_y, tau_x), axis=(0, 1))
    
    # Convert to bytes for compression
    ref_bytes = (ref_img * 255).astype(np.uint8).tobytes()
    shifted_bytes = (shifted * 255).astype(np.uint8).tobytes()
    
    score = compute_ncd(ref_bytes, shifted_bytes)
    
    return InformativenessReport(
        score=score,
        tau_star_x=tau_x,
        tau_star_y=tau_y,
        flag_low_confidence=(score < threshold)
    )
