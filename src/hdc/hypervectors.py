"""Bipolar Hyperdimensional Computing (HDC) encoder for patch-level embeddings.

Implements Stage 1 of the ATLAS pipeline: encoding image patches into
high-dimensional bipolar hypervectors ({0, 1}^D) using position-value
binding and majority-vote bundling.

The encoder uses row-wise accumulation to avoid materializing the full
(N, H, W, D) tensor, keeping peak memory at O(N * W * D) per row.

References:
    Neubert et al., "Hyperdimensional Computing as a Framework for
    Systematic Aggregation of Image Descriptors", CVPR 2021.
"""

import torch
import numpy as np
from typing import Tuple

# Force CPU to avoid CUDA driver issues on systems with prior OOM crashes.
# For GPU acceleration, change to: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_DEVICE = torch.device('cpu')


class BipolarHypervectorSpace:
    """Bipolar hypervector space for spatial-aware patch encoding.

    Encodes (H, W) image patches into D-dimensional bipolar vectors using:
      - Value HVs: random codebook for quantized intensity bins
      - Position HVs: permutation-generated spatial encoders for (x, y)
      - Binding: element-wise XOR (equivalent to multiplication in {-1,+1})
      - Bundling: majority vote across bound pixel vectors

    Args:
        d: Hypervector dimensionality (default 2048).
        seed: Random seed for reproducible codebook generation.
    """

    def __init__(self, d: int = 2048, seed: int = 42):
        self.d = d
        torch.manual_seed(seed)

        # Intensity codebook: 16 quantization bins
        self.num_bins = 16
        self.hv_val = torch.randint(0, 2, (self.num_bins, d), dtype=torch.int8, device=_DEVICE)

        # Base position hypervectors
        hv_x_base = torch.randint(0, 2, (d,), dtype=torch.int8, device=_DEVICE)
        hv_y_base = torch.randint(0, 2, (d,), dtype=torch.int8, device=_DEVICE)

        # Random permutations for spatial encoding (deterministic from seed)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed + 1)
        self.perm_x = torch.randperm(d, generator=gen).to(_DEVICE)
        gen.manual_seed(seed + 2)
        self.perm_y = torch.randperm(d, generator=gen).to(_DEVICE)

        # Pre-compute position HV cache up to max_size
        self.max_size = 200
        self.hv_x_cache = torch.zeros((self.max_size, d), dtype=torch.int8, device=_DEVICE)
        self.hv_y_cache = torch.zeros((self.max_size, d), dtype=torch.int8, device=_DEVICE)
        self.hv_x_cache[0] = hv_x_base
        self.hv_y_cache[0] = hv_y_base
        for i in range(1, self.max_size):
            self.hv_x_cache[i] = self.hv_x_cache[i - 1][self.perm_x]
            self.hv_y_cache[i] = self.hv_y_cache[i - 1][self.perm_y]

    def bundle(self, hvs: list) -> np.ndarray:
        """Majority-vote bundling over a list of bipolar hypervectors.

        Maps {0,1} → {-1,+1}, sums, then thresholds back to {0,1}.
        Returns numpy int8 array of shape (D,).
        """
        if not hvs:
            return np.zeros(self.d, dtype=np.int8)
        tensors = []
        for hv in hvs:
            if isinstance(hv, np.ndarray):
                tensors.append(torch.from_numpy(hv).to(_DEVICE))
            else:
                tensors.append(hv)
        stacked = torch.stack(tensors)
        polar = stacked.to(torch.int16) * 2 - 1
        summed = polar.sum(dim=0)
        return (summed > 0).to(torch.int8).cpu().numpy()

    def cosine_similarity_batch(self, a: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        """Batch Hamming-based cosine similarity between a reference HV and N candidates.

        Converts normalized Hamming distance to cosine-like similarity in [-1, 1].
        """
        a_t = torch.from_numpy(a).to(_DEVICE)
        b_t = torch.from_numpy(b_batch).to(_DEVICE)
        hamming_dists = (a_t != b_t).sum(dim=1).float()
        sims = 1.0 - (2.0 * hamming_dists / self.d)
        return sims.cpu().numpy()

    def encode_patch(self, patch: np.ndarray) -> np.ndarray:
        """Encode a single (H, W) grayscale patch into a D-dimensional bipolar HV."""
        return self.encode_batch_rowwise(patch[np.newaxis, :, :])[0]

    def encode_batch_rowwise(self, patches: np.ndarray, chunk_size: int = 256) -> np.ndarray:
        """Batch-encode multiple patches using row-wise accumulation.

        Iterates over H rows instead of materializing the full (N, H, W, D)
        tensor. Each row step allocates only (chunk_size, W, D), keeping
        peak memory bounded at O(chunk_size * W * D).

        Args:
            patches: (N, H, W) float32 array with values in [0, 1].
            chunk_size: Number of patches to process per memory chunk.

        Returns:
            (N, D) int8 numpy array of bipolar hypervectors.
        """
        N, H, W = patches.shape
        all_results = []

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = patches[start:end]
            C = chunk.shape[0]

            # Quantize intensities to bin indices
            chunk_t = torch.from_numpy(chunk.astype(np.float32)).to(_DEVICE)
            quantized = (chunk_t * (self.num_bins - 1)).clamp(0, self.num_bins - 1).long()

            # Accumulator for bundling across all pixels
            accum = torch.zeros(C, self.d, dtype=torch.int32, device=_DEVICE)
            hx = self.hv_x_cache[:W]

            # Row-wise accumulation: only (C, W, D) resident at any time
            for row in range(H):
                row_vals = self.hv_val[quantized[:, row, :]]         # (C, W, D)
                spatial = torch.bitwise_xor(self.hv_y_cache[row], hx)  # (W, D)
                bound = torch.bitwise_xor(row_vals, spatial.unsqueeze(0))
                polar = bound.to(torch.int16) * 2 - 1
                accum += polar.sum(dim=1).to(torch.int32)
                del row_vals, spatial, bound, polar

            result = (accum > 0).to(torch.int8).cpu().numpy()
            all_results.append(result)
            del chunk_t, quantized, accum

        return np.concatenate(all_results, axis=0)
