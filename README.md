# ATLAS / Drift-Sense

**Algebraic-Topological Localization via Adic Search**  
*A Cross-Magnification Navigation-Error Recovery Framework for the Applied Materials "Drift-Sense" Problem Statement.*

---

## 1. Problem Restatement and Formal Definition

Let $R \in \mathbb{R}^{1000\times1000}$ be the grayscale reference image (100x magnification) and let $S \in \mathbb{R}^{1000\times1000}$ be the grayscale wide-search image (10x magnification). The two captures are independent acquisitions of the same physical die, carrying independent sensor noise realizations. There exists an unknown similarity transform:

$$ g = (s,\theta,\mathbf{t}) \in \mathrm{Sim}(2), \qquad s \in [0.09,0.11], \ \ \theta \in [-2^\circ,2^\circ], \ \ \mathbf{t}\in\mathbb{R}^2 $$

such that:

$$ R(\mathbf{x}) \approx S\big(g(\mathbf{x})\big) + n(\mathbf{x}), \qquad \mathbf{x}\in[0,1000)^2 $$

where $n$ denotes the independent acquisition noise of $S$. The deliverable is the pixel coordinate $\mathbf{c}^\ast = g(\mathbf{x}_{\text{center of }R})$ in $S$-pixel space, with ties among statistically indistinguishable candidates broken by choosing the one closest to the center of $S$.

**Two key difficulties:**
1. **Global periodicity:** DRAM and FinFET arrays are translationally periodic. A naive similarity score will have many near-tied local maxima.
2. **Cross-magnification:** The transform is a genuine similarity transform (scale, rotation, translation), not an unconstrained deep-learning correspondence problem.

---

## 2. ATLAS System Architecture

ATLAS decomposes the task into a pipeline of rigorously defined mathematical subproblems.

### Stage 0: Kolmogorov/NCD Informativeness Gate
Before any search is attempted, ATLAS scores the intrinsic ambiguity of the reference patch. We use Normalized Compression Distance (NCD):

$$ \mathrm{NCD}(x,y) = \frac{C(xy) - \min\{C(x),C(y)\}}{\max\{C(x),C(y)\}} $$

We estimate the dominant lattice period $\tau^\ast = \arg\max_{\tau} \mathrm{Autocorr}(R,\tau)$. The informativeness score is defined as the patch compared against its own shifted copy:

$$ I(R) = \mathrm{NCD}\big(R,\ \mathrm{shift}(R,\tau^\ast)\big) $$

If $I(R)$ is low, the patch is perfectly periodic and ambiguous, raising an *a priori* low-confidence flag before search begins.

### Stage 1: Hyperdimensional Multi-Scale Shortlist
To avoid exhaustive $O(N^2)$ pixel sliding, we build a shortlist using Vector Symbolic Architectures (Hyperdimensional Computing).
*   **Bipolar space**: $\{-1,+1\}^D$ with $D \approx 10,000$.
*   **Binding**: Elementwise XOR ($\otimes$).
*   **Bundling**: Elementwise majority vote.
*   **Patch Encoding**: $\mathrm{HV}(P) = \mathrm{sign}\Big(\sum_{i,j} \mathrm{HV}_{\mathrm{val}}\big(q(P[i,j])\big)\otimes \mathrm{HV}_x(i)\otimes \mathrm{HV}_y(j)\Big)$
*   **Rotation Tolerance**: Bundled directly without searching: $\mathrm{HV}^{\mathrm{rot}}(R) = \mathrm{sign}\big(\sum_{\theta'\in\{-2^\circ,0,2^\circ\}} \mathrm{HV}(\mathrm{rotate}(R,\theta'))\big)$.

This yields a candidate shortlist $\mathcal{K}$ in near-linear time using bit-parallel array operations.

### Stage 2: $4$-adic Quadtree Ultrametric Pruning
We refine the shortlist bounds using the $p$-adic numbers.
*   **Morton Code**: A depth-$d$ Morton address $a = (a_1 a_2 \dots a_d)$ is identified as a truncated $4$-adic integer:

    $$ A = \sum_{i=1}^{d} a_i\,4^{\,i-1} \in \mathbb{Z}/4^{d}\mathbb{Z} \subset \mathbb{Z}_4 $$

*   **Ultrametric**: The $4$-adic valuation $v_4(A-A')$ equals the length of the common Z-order prefix. The induced distance is $d_4(A,A') = 4^{-v_4(A-A')}$.

This ultrametric corresponds exactly to spatial nesting distance in a quadtree. We estimate the image Lipschitz constant $L$ and use $Score(\nu) + L \cdot 4^{-k} \cdot \text{size}$ to prune subtrees in $O(N \log N)$ expected time, providing a mathematically provable branch-and-bound discard criterion.

### Stage 3: Fourier-Mellin Coarse Pose Estimate
For each surviving candidate tile $T_i$, we compute the classical Fourier-Mellin invariant descriptor pair.
Magnitude spectra $|\mathcal{F}(R)|$ and $|\mathcal{F}(T_i)|$ are remapped to log-polar coordinates $(\rho,\phi) = (\log r, \arctan(y/x))$. Phase correlation recovers scale $\hat s_i$ and rotation $\hat\theta_i$. A second phase correlation on the corrected images recovers translation $\hat{\mathbf{t}}_i$. This explicitly measures the 10x scale ratio instead of assuming it.

### Stage 4: Conformal Geometric Algebra Joint Refinement
We refine $(s,\theta,t_x,t_y)$ jointly using a Matrix Lie Group implementation of the $Sim(2)$ similarity group.
A translation, rotation, and uniform dilation are composed multiplicatively on the manifold. We perform gradient descent along the bivector generators of the Lie algebra ($\mathfrak{sim}(2)$):

$$ V_{k+1} = \exp\!\big(-\eta\,\nabla_{\mathfrak{g}} E(V_k)\big)\,V_k $$

where the update uses the true matrix exponential (`scipy.linalg.expm`). This avoids the coordinate singularities (gimbal lock) of naive alternating line searches and achieves true sub-pixel accuracy.

### Stage 5: Persistent-Homology Topological Tie-Break
If candidates remain tied within measurement noise ($\mathcal{C}^\ast$), correlation scores are insufficient. A periodic lattice can have identical intensity statistics but differ in fine connectivity (a missing contact loop).
We build the cubical sublevel-set filtration $\{P \le t\}$ and extract the $1$-dimensional persistence diagrams (loops) $D_1$. The topological distance to the reference is the bottleneck distance:

$$ d_{\mathrm{topo}}(P) = d_B\big(D_1(R), D_1(P)\big) $$

### Stage 6: Fused Confidence
ATLAS reports a single repeatable confidence scalar per prediction:

$$ \mathrm{Conf} = w_1\,\mathrm{NCC}^\ast + w_2\,\cos_{\mathrm{HDC}} + w_3\big(1 - \widehat d_{\mathrm{topo}}\big) - w_4\big(1 - I(R)\big) $$

This guarantees that a prediction against a low-informativeness periodic reference is reported with appropriately low confidence.

---

## Usage & Execution

### Setup
```bash
pip install -r requirements.txt
```

### 1. Generating Synthetic Data
Generates accurate DRAM ($6F^2$/$8F^2$) and FinFET structures with strict independent mixed Poisson-Gaussian noise models, satisfying the 30% augmentation rubric requirement.
```bash
python generate_dataset.py --arch both --num-pairs 30 --out-dir results/dataset
```

### 2. Running Localization (Scored CLI)
The standalone CLI script required by the rubric for headless evaluation.
```bash
# Basic Output (x,y)
python localize.py --reference path/to/ref.png --search path/to/search.png

# Full Details Output
python localize.py --reference path/to/ref.png --search path/to/search.png --json
```

### 3. Evaluating Batch Metrics
Runs the entire generated manifest and reports $1px/2px/4px/5px$ pass rates and errors.
```python
# (Using the evaluation runner in Python)
from src.eval.runner import EvaluationRunner
runner = EvaluationRunner("results/dataset/manifest.csv", "results/eval")
runner.run_all()
```

### 4. Running the GUI Console
Starts the PyQt6 visualization dashboard.
```bash
python src/gui/app.py
```
