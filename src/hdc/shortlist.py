"""Multi-scale HDC shortlist computation for candidate position ranking.

Implements the sliding-window hyperdimensional shortlisting described in
Stage 1 of the ATLAS paper. For each scale hypothesis, the reference patch
is encoded as a rotation-tolerant bundled hypervector, and candidate
positions in the search image are ranked by Hamming-based cosine similarity.

The implementation uses numpy stride_tricks for zero-copy sliding windows
and batch-encodes all patches via the row-wise accumulation strategy in
BipolarHypervectorSpace to control memory usage.
"""

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
    stride: int = 15,
    top_k: int = 20
) -> List[Tuple[float, float, float, float]]:
    """Compute a ranked shortlist of candidate positions via HDC similarity.

    For each (scale, rotation) hypothesis, the reference is warped, encoded
    into a bundled hypervector, and compared against all sliding-window
    patches extracted from the search image at the given stride.

    Args:
        ref_img: (H_r, W_r) float32 reference image in [0, 1].
        search_img: (H_s, W_s) float32 search image in [0, 1].
        hv_space: Pre-initialized BipolarHypervectorSpace instance.
        scale_hypotheses: List of scale factors to evaluate (e.g. [0.095, 0.1, 0.105]).
        rotation_hypotheses: List of rotation angles in degrees (e.g. [-2, 0, 2]).
        stride: Sliding window step size in pixels.
        top_k: Number of top candidates to return.

    Returns:
        List of (center_x, center_y, scale, similarity) tuples, sorted by
        descending similarity, truncated to top_k.
    """
    rh, rw = ref_img.shape
    sh, sw = search_img.shape
    all_candidates = []

    for s in scale_hypotheses:
        scaled_rw = int(rw * s)
        scaled_rh = int(rh * s)

        if scaled_rh > sh or scaled_rw > sw or scaled_rh < 4 or scaled_rw < 4:
            continue

        # Encode rotated references and bundle into a single rotation-tolerant HV
        rot_hvs = []
        for theta in rotation_hypotheses:
            M = cv2.getRotationMatrix2D((rw / 2.0, rh / 2.0), theta, s)
            M[0, 2] += (scaled_rw / 2.0) - (rw / 2.0)
            M[1, 2] += (scaled_rh / 2.0) - (rh / 2.0)
            rot_ref = cv2.warpAffine(ref_img, M, (scaled_rw, scaled_rh))
            rot_hvs.append(hv_space.encode_patch(rot_ref))

        bundled_ref_hv = hv_space.bundle(rot_hvs)

        # Extract all sliding-window patches via stride_tricks (zero-copy)
        view = np.lib.stride_tricks.sliding_window_view(search_img, (scaled_rh, scaled_rw))
        view_strided = view[::stride, ::stride, :, :]
        H_out, W_out, _, _ = view_strided.shape

        patches_batch = np.ascontiguousarray(view_strided.reshape(-1, scaled_rh, scaled_rw))
        if patches_batch.shape[0] == 0:
            continue

        # Batch encode and compute similarities
        batch_hvs = hv_space.encode_batch_rowwise(patches_batch, chunk_size=256)
        sims = hv_space.cosine_similarity_batch(bundled_ref_hv, batch_hvs)

        # Retain only the top candidates per scale to limit list growth
        if len(sims) > top_k * 3:
            top_indices = np.argpartition(sims, -top_k * 3)[-top_k * 3:]
        else:
            top_indices = np.arange(len(sims))

        y_indices, x_indices = np.unravel_index(top_indices, (H_out, W_out))
        orig_y = y_indices * stride
        orig_x = x_indices * stride
        center_x = orig_x + (scaled_rw / 2.0)
        center_y = orig_y + (scaled_rh / 2.0)

        for i in range(len(top_indices)):
            all_candidates.append((center_x[i], center_y[i], s, sims[top_indices[i]]))

    all_candidates.sort(key=lambda c: c[3], reverse=True)
    return all_candidates[:top_k]
