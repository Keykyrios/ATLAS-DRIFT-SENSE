# ATLAS / Drift-Sense References

This document traces every mathematical and config-driven choice in the pipeline to a specific published source, as required by the problem statement rubric (30% augmentation score requirement).

## Stage 0: Informativeness Gate (NCD)
*   **Li et al. (2004) & Cilibrasi & Vitányi (2005)**: The use of Normalized Compression Distance (NCD) as a proxy for Kolmogorov-complexity-based information distance. We use `zlib` to implement the compressor.
    *   *Reference*: Li, M., Chen, X., Li, X., Ma, B., & Vitányi, P. M. (2004). The similarity metric. *IEEE Transactions on Information Theory*.

## Stage 1: HDC Multi-Scale Shortlist
*   **Neubert et al. (2021)**: The use of hyperdimensional computing (HDC) and Vector Symbolic Architectures (VSA) for image descriptor aggregation, applying binding ($\otimes$) for spatial constraints and bundling ($\oplus$) for rotation tolerance.
    *   *Reference*: Neubert, P., Schubert, S., & Protzel, P. (2021). Hyperdimensional Computing as a Framework for Systematic Aggregation of Image Descriptors. *CVPR*.

## Stage 2: Quadtree 4-Adic Pruning
*   **Kothuri et al. (2002)**: Morton (Z-order) codes and quadtree spatial indexing. We explicitly map the depth-$d$ Morton code to a truncated 4-adic integer to build the $4$-adic ultrametric pruning branch-and-bound.
    *   *Reference*: Kothuri, R. K. V., Ravada, S., & Abugov, D. (2002). Quadtree and R-tree indexes in Oracle Spatial. *ACM SIGMOD*.

## Stage 3: Fourier-Mellin Coarse Pose
*   **Reddy & Chatterji (1996)**: Log-polar phase correlation for scale, rotation, and translation estimation.
    *   *Reference*: Reddy, B. S., & Chatterji, B. N. (1996). An FFT-based technique for translation, rotation, and scale-invariant image registration. *IEEE Transactions on Image Processing*.

## Stage 4: Conformal Geometric Algebra Refinement
*   **Hitzer & Sangwine (2019)**: The versor representation of the similarity group $Sim(2)$ in Conformal Geometric Algebra ($V = T \cdot R \cdot D$).
    *   *Reference*: Hitzer, E., & Sangwine, S. J. (2019). Foundations of Conic Conformal Geometric Algebra and Compact Versors for Rotation, Translation and Scaling. *Advances in Applied Clifford Algebras*.

## Stage 5: Topological Tie-Break (Persistent Homology)
*   **Edelsbrunner & Harer (2010)**: Computational topology, specifically cubical sublevel-set filtration and bottleneck distance for $1$-dimensional persistence diagrams (loops).
    *   *Reference*: Edelsbrunner, H., & Harer, J. (2010). Computational Topology: An Introduction. *American Mathematical Society*.

## Dataset Generation (Mixed Poisson-Gaussian SEM Noise)
*   **SEM Noise Physics Literature**: The mixed Poisson-Gaussian model ($Y = X + n_p(X) + n_g$) captures the primary/secondary electron count shot noise and detector floor.
    *   *References*:
        1. "Scanning Electron Microscope Image Signal-to-Noise Ratio Monitoring", HAL-01051309.
        2. "Bayesian Deconvolution of Scanning Electron Microscopy Images Using Point-Spread Function Estimation and Non-local Regularization", arXiv:1810.09739.

## DRAM/FinFET Array Layout Geometry
*   **DRAM Patent Structural Characteristics**: The $6F^2$ and $8F^2$ periodic word-line and bit-line crossed layouts.
    *   *References*:
        1. US Patent 6,768,663 B2, "Semiconductor device array having dense memory cell array and hierarchical bit line scheme."
        2. US Patent 6,097,621, "Memory cell array architecture for random access memory device."
