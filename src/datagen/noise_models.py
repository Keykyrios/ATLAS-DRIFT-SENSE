import numpy as np

def apply_mixed_poisson_gaussian_noise(image: np.ndarray, a: float, b: float, seed: int) -> np.ndarray:
    """
    Applies the mixed Poisson-Gaussian SEM noise model.
    Y ~ N(X, aX + b^2)
    This formulation matches the Gaussian approximation of the SEM noise physics
    where 'a' scales the Poisson shot-noise (signal dependent) and 'b' is the 
    Gaussian detector floor (signal independent).
    
    Args:
        image: Original image in [0, 1] range.
        a: Poisson scale parameter.
        b: Gaussian floor std dev parameter.
        seed: Random seed for EXACT independent noise generation.
    Returns:
        Noisy image in [0, 1] range.
    """
    rng = np.random.default_rng(seed)
    
    # Scale image to some nominal electron count, say max 255 for standard 8-bit equivalent
    X = image * 255.0
    
    # Variance map: aX + b^2
    variance = a * X + (b ** 2)
    variance = np.clip(variance, 0, None)
    std_dev = np.sqrt(variance)
    
    # Generate Gaussian noise with signal-dependent variance
    noise = rng.normal(loc=0.0, scale=std_dev, size=image.shape)
    
    noisy_img = X + noise
    # Normalize back to [0, 1]
    noisy_img = noisy_img / 255.0
    return np.clip(noisy_img, 0, 1)
