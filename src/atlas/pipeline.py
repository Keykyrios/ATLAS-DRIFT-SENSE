import time
import numpy as np
import cv2
import yaml
import os

from .types import ATLASResult
from .stage0_informativeness import run_stage0
from .stage1_hdc_shortlist import run_stage1
from .stage2_quadtree_prune import run_stage2
from .stage3_fourier_mellin import run_stage3
from .stage4_cga_refine import run_stage4
from .stage5_topological_tiebreak import run_stage5
from .stage6_confidence import run_stage6

class ATLASPipeline:
    def __init__(self, config_path: str = "configs/default.yaml"):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "stage0": {}, "stage1": {}, "stage2": {}, 
                "stage4": {}, "stage5": {}, "stage6": {}
            }

    def process(self, ref_img: np.ndarray, search_img: np.ndarray) -> ATLASResult:
        """
        Runs the full 7-stage ATLAS pipeline on a pair of images.
        """
        # Ensure grayscale and normalized
        if len(ref_img.shape) == 3:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        if len(search_img.shape) == 3:
            search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
            
        if ref_img.dtype == np.uint8:
            ref_img = ref_img.astype(np.float32) / 255.0
        if search_img.dtype == np.uint8:
            search_img = search_img.astype(np.float32) / 255.0
            
        t_start_total = time.perf_counter()
        timings = {}
        
        # Stage 0: Informativeness
        t0 = time.perf_counter()
        info_report = run_stage0(ref_img, self.config.get('stage0', {}))
        timings['stage0'] = (time.perf_counter() - t0) * 1000
        
        # Stage 1: HDC Shortlist
        t1 = time.perf_counter()
        candidates = run_stage1(ref_img, search_img, self.config.get('stage1', {}))
        
        # Save HDC scores for Stage 6
        hdc_map = {(c.x, c.y): c.score for c in candidates}
        timings['stage1'] = (time.perf_counter() - t1) * 1000
        
        # Stage 2: Quadtree Pruning
        t2 = time.perf_counter()
        survivors_st2 = run_stage2(search_img, candidates, self.config.get('stage2', {}))
        timings['stage2'] = (time.perf_counter() - t2) * 1000
        
        # Stage 3: Fourier-Mellin
        t3 = time.perf_counter()
        poses_st3 = run_stage3(ref_img, search_img, survivors_st2, self.config.get('stage3', {}))
        timings['stage3'] = (time.perf_counter() - t3) * 1000
        
        # Stage 4: CGA Refinement
        t4 = time.perf_counter()
        refined_cands_st4 = run_stage4(ref_img, search_img, poses_st3, self.config.get('stage4', {}))
        timings['stage4'] = (time.perf_counter() - t4) * 1000
        
        # Stage 5: Topological Tie-break
        t5 = time.perf_counter()
        tied_set_st5 = run_stage5(ref_img, search_img, refined_cands_st4, self.config.get('stage5', {}))
        timings['stage5'] = (time.perf_counter() - t5) * 1000
        
        # Stage 6: Confidence
        t6 = time.perf_counter()
        t_end_total = time.perf_counter()
        total_ms = (t_end_total - t_start_total) * 1000
        
        result = run_stage6(
            search_img, tied_set_st5, info_report, 
            self.config.get('stage6', {}), hdc_map, total_ms, timings
        )
        timings['stage6'] = (time.perf_counter() - t6) * 1000
        
        return result
