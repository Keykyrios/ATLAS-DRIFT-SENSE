import cv2
import numpy as np
import scipy.linalg

class CGA2DSim:
    """
    Rigorous Sim(2) Lie Group implementation for joint similarity refinement.
    Implements the versor composition V = T * R * D for similarity transforms.
    Performs gradient descent along the bivector tangent space.
    """
    def __init__(self, ref_img: np.ndarray, search_img: np.ndarray):
        self.ref_img = ref_img.astype(np.float32)
        self.search_img = search_img.astype(np.float32)
        self.h, self.w = self.ref_img.shape
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        
        # Define Sim(2) generators
        self.G_s = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        self.G_th = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64)
        self.G_tx = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        self.G_ty = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float64)

    def _build_matrix(self, s: float, theta: float, tx: float, ty: float) -> np.ndarray:
        """Builds initial Sim(2) matrix. Note: theta in degrees."""
        th_rad = np.deg2rad(theta)
        # To rotate around center, we translate to origin, rotate/scale, translate back, then add tx, ty
        T_center_inv = np.array([[1, 0, -self.cx], [0, 1, -self.cy], [0, 0, 1]])
        T_center = np.array([[1, 0, self.cx], [0, 1, self.cy], [0, 0, 1]])
        
        R_S = np.array([
            [s * np.cos(th_rad), -s * np.np.sin(th_rad), 0],
            [s * np.sin(th_rad), s * np.cos(th_rad), 0],
            [0, 0, 1]
        ])
        
        T_trans = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        
        return T_trans @ T_center @ R_S @ T_center_inv

    def _warp_image(self, M: np.ndarray) -> np.ndarray:
        # M maps from ref to search. We need M_inv to pull search pixels back to ref.
        # But wait, the standard OpenCV affine warp takes a 2x3 matrix mapping ref->search
        # and warpAffine computes the inverse mapping internally if we pass WARP_INVERSE_MAP.
        # Actually it's easier to just invert M explicitly to be sure.
        M_inv = np.linalg.inv(M)
        M_cv2 = M_inv[:2, :]
        warped = cv2.warpAffine(self.search_img, M_cv2, (self.w, self.h), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped

    def _compute_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        res = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        return res[0][0]

    def refine(self, init_s: float, init_theta: float, init_tx: float, init_ty: float, 
               max_iter: int = 20, lr: float = 0.05, tol: float = 1e-4) -> tuple:
        """
        Gradient descent directly on the Sim(2) Lie algebra.
        """
        # M maps points in reference image to points in search image
        # Note: init_s is the scale factor from ref -> search
        # E.g. if ref is 100x100 and it appears at 100x100 in search, s = 1.0.
        M = self._build_matrix(1.0/init_s, init_theta, init_tx, init_ty)
        
        prev_ncc = -1.0
        converged = False
        
        for i in range(max_iter):
            warped = self._warp_image(M)
            ncc = self._compute_ncc(self.ref_img, warped)
            
            if np.abs(ncc - prev_ncc) < tol:
                converged = True
                break
                
            # Finite differences on the Lie algebra manifold
            delta = 1e-3
            
            # Step in each generator direction
            M_s = M @ scipy.linalg.expm(delta * self.G_s)
            grad_s = (self._compute_ncc(self.ref_img, self._warp_image(M_s)) - ncc) / delta
            
            M_th = M @ scipy.linalg.expm(delta * self.G_th)
            grad_th = (self._compute_ncc(self.ref_img, self._warp_image(M_th)) - ncc) / delta
            
            M_tx = M @ scipy.linalg.expm(delta * self.G_tx)
            grad_tx = (self._compute_ncc(self.ref_img, self._warp_image(M_tx)) - ncc) / delta
            
            M_ty = M @ scipy.linalg.expm(delta * self.G_ty)
            grad_ty = (self._compute_ncc(self.ref_img, self._warp_image(M_ty)) - ncc) / delta
            
            # Form the Lie algebra gradient step (we maximize NCC)
            # Preconditioning: Translation gradients in pixel units are much smaller than 
            # rotation/scale gradients (radians/proportions). We scale tx, ty steps by 10.
            step_algebra = lr * (grad_s * self.G_s + grad_th * self.G_th + 
                                 (grad_tx * 10.0) * self.G_tx + (grad_ty * 10.0) * self.G_ty)
            
            # Group exponential map update
            M = M @ scipy.linalg.expm(step_algebra)
            
            prev_ncc = ncc
            
        final_warped = self._warp_image(M)
        final_ncc = self._compute_ncc(self.ref_img, final_warped)
        
        # Extract refined parameters from final matrix M
        # M maps from ref -> search
        s_refined = 1.0 / np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        th_refined = np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))
        
        # We want the translation of the center
        center_homog = np.array([self.cx, self.cy, 1.0])
        mapped_center = M @ center_homog
        tx_refined = mapped_center[0]
        ty_refined = mapped_center[1]
        
        return s_refined, th_refined, tx_refined, ty_refined, final_ncc, converged
