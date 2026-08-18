import numpy as np
from src.datagen.noise_models import apply_mixed_poisson_gaussian_noise

def test_independent_noise_streams():
    """
    Asserts noise arrays on Reference and Search are NOT identical,
    validating the independent noise rubric requirement.
    """
    img = np.ones((100, 100)) * 0.5
    
    seed_ref = 42
    seed_search = 43 # MUST be different
    
    noisy_ref = apply_mixed_poisson_gaussian_noise(img, a=1.0, b=2.0, seed=seed_ref)
    noisy_search = apply_mixed_poisson_gaussian_noise(img, a=1.0, b=2.0, seed=seed_search)
    
    # Residuals
    res_ref = noisy_ref - img
    res_search = noisy_search - img
    
    # Ensure they are not exactly the same
    assert not np.allclose(res_ref, res_search)
    
    # Cross correlation should be near zero (independent)
    corr = np.corrcoef(res_ref.flatten(), res_search.flatten())[0, 1]
    assert abs(corr) < 0.1 # Very low correlation
