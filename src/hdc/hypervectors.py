import numpy as np
from typing import Tuple

class BipolarHypervectorSpace:
    """
    Implements bipolar HDC using packed bit arrays for speed.
    Supports parallel batch encoding.
    (Bit 1 = +1, Bit 0 = -1)
    """
    def __init__(self, d: int = 10000, seed: int = 42):
        self.d = d
        self.rng = np.random.default_rng(seed)
        
        # Bins for intensity (16 bins)
        self.num_bins = 16
        self.hv_val = self._generate_random_hvs(self.num_bins)
        
        # Position base HVs
        self.hv_x_base = self._generate_random_hvs(1)[0]
        self.hv_y_base = self._generate_random_hvs(1)[0]
        
        # We precompute a fixed random permutation for translation.
        self.perm_x = self.rng.permutation(self.d)
        self.perm_y = self.rng.permutation(self.d)
        
        # Pre-generate max spatial HVs (assuming max patch size ~200)
        self.max_size = 200
        self.hv_x_cache = np.zeros((self.max_size, self.d), dtype=np.int8)
        self.hv_y_cache = np.zeros((self.max_size, self.d), dtype=np.int8)
        
        self.hv_x_cache[0] = self.hv_x_base
        self.hv_y_cache[0] = self.hv_y_base
        
        for i in range(1, self.max_size):
            self.hv_x_cache[i] = self.hv_x_cache[i-1][self.perm_x]
            self.hv_y_cache[i] = self.hv_y_cache[i-1][self.perm_y]

    def _generate_random_hvs(self, count: int) -> np.ndarray:
        return self.rng.integers(0, 2, size=(count, self.d), dtype=np.int8)
        
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Binding is XOR for {0, 1} representation of {-1, 1}."""
        return np.bitwise_xor(a, b)
        
    def bundle(self, hvs: list) -> np.ndarray:
        """Bundling is majority vote over a list."""
        if not hvs:
            return np.zeros(self.d, dtype=np.int8)
        sum_vec = np.sum([hv * 2 - 1 for hv in hvs], axis=0)
        return (sum_vec > 0).astype(np.int8)
        
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Normalized Hamming distance converted to cosine-like similarity."""
        hamming_dist = np.count_nonzero(a != b)
        return 1.0 - (2.0 * hamming_dist / self.d)
        
    def cosine_similarity_batch(self, a: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        """Batch Hamming distance computation."""
        # a: (D,), b_batch: (N, D)
        hamming_dists = np.count_nonzero(a != b_batch, axis=1)
        return 1.0 - (2.0 * hamming_dists / self.d)
        
    def encode_patch(self, patch: np.ndarray) -> np.ndarray:
        # Fallback to batch encoding
        return self.encode_batch(patch[np.newaxis, :, :])[0]
        
    def encode_batch(self, patches: np.ndarray) -> np.ndarray:
        """
        Parallel HDC encoding via array broadcasting.
        patches: (N, H, W) float array in [0, 1]
        Returns: (N, D) bipolar hypervectors
        """
        N, H, W = patches.shape
        if H > self.max_size or W > self.max_size:
            raise ValueError(f"Patch size exceeds precomputed spatial cache ({self.max_size})")
            
        # 1. Quantize all patches at once -> (N, H, W)
        quantized = (patches * (self.num_bins - 1)).astype(np.int32)
        quantized = np.clip(quantized, 0, self.num_bins - 1)
        
        # 2. Lookup value HVs -> (N, H, W, D)
        val_hvs = self.hv_val[quantized]
        
        # 3. Create spatial binding map -> (H, W, D)
        hx = self.hv_x_cache[:W] # (W, D)
        hy = self.hv_y_cache[:H] # (H, D)
        # Broadcasting: hy[:, None, :] ^ hx[None, :, :] -> (H, W, D)
        spatial_map = self.bind(hy[:, np.newaxis, :], hx[np.newaxis, :, :])
        
        # 4. Bind value HVs with spatial map -> (N, H, W, D)
        bound_hvs = self.bind(val_hvs, spatial_map[np.newaxis, :, :, :])
        
        # 5. Bundle (sum across H and W) -> (N, D)
        # Map {0, 1} to {-1, 1} for sum
        bound_polar = bound_hvs.astype(np.int32) * 2 - 1
        sum_vec = np.sum(bound_polar, axis=(1, 2))
        
        # Map back to {0, 1}
        return (sum_vec > 0).astype(np.int8)
