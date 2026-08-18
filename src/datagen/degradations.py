import cv2
import numpy as np

def apply_psf_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Applies a separable Gaussian Point Spread Function (PSF) to mimic SEM beam spot size.
    """
    if sigma <= 0:
        return image
    
    ksize = int(2 * np.ceil(2 * sigma) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def apply_edge_brightening(image: np.ndarray, boost_factor: float) -> np.ndarray:
    """
    Mimics secondary-electron edge-contrast effect. 
    Boosts intensity where gradient magnitude is high.
    """
    if boost_factor <= 1.0:
        return image
        
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # Normalize gradient magnitude
    if np.max(grad_mag) > 0:
        grad_mag = grad_mag / np.max(grad_mag)
        
    # Add scaled gradient to image
    brightened = image + (grad_mag * (boost_factor - 1.0))
    return np.clip(brightened, 0, 1)
