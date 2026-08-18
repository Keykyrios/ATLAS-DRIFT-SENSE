import numpy as np
from typing import List, Tuple
from .types import Candidate, InformativenessReport, ATLASResult

def run_stage6(
    search_img: np.ndarray, 
    tied_set: List[Candidate], 
    info_report: InformativenessReport, 
    config: dict,
    hdc_score_map: dict,
    total_time_ms: float,
    stage_timings: dict
) -> ATLASResult:
    """
    Stage 6: Fused Confidence and Report
    Computes final fused score and applies closest-to-center fallback if needed.
    """
    if not tied_set:
        return ATLASResult(-1, -1, 0.0, True, total_time_ms, stage_timings, False)
        
    w_ncc = config.get('w_ncc', 0.4)
    w_hdc = config.get('w_hdc', 0.2)
    w_topo = config.get('w_topo', 0.3)
    w_info = config.get('w_info', 0.1)
    
    # Check for topological tie (if d_topo is identical for top candidates)
    best_topo = tied_set[0].d_topo
    topo_tied = [c for c in tied_set if abs(c.d_topo - best_topo) < 1e-5]
    
    tie_break_applied = False
    
    if len(topo_tied) > 1:
        # Fall back to closest-to-center rule
        tie_break_applied = True
        cy, cx = search_img.shape[0] / 2, search_img.shape[1] / 2
        topo_tied.sort(key=lambda c: np.sqrt((c.x - cx)**2 + (c.y - cy)**2))
        
    final_cand = topo_tied[0]
    
    # Calculate confidence
    # Conf = w1*NCC* + w2*cos_HDC + w3*(1 - d_topo) - w4*(1 - I(R))
    # We bounded elements to [0, 1] approximately.
    
    ncc = final_cand.score
    hdc = hdc_score_map.get((final_cand.x, final_cand.y), final_cand.score) # Fallback to NCC if not exact map
    d_topo_norm = final_cand.d_topo_norm
    info_score = info_report.score
    
    conf = (w_ncc * ncc) + (w_hdc * hdc) + (w_topo * (1.0 - d_topo_norm)) - (w_info * (1.0 - info_score))
    # Bound to [0, 1]
    conf = max(0.0, min(1.0, conf))
    
    return ATLASResult(
        x=final_cand.x,
        y=final_cand.y,
        confidence=conf,
        is_low_informativeness=info_report.flag_low_confidence,
        runtime_ms=total_time_ms,
        stage_timings=stage_timings,
        tie_break_applied=tie_break_applied
    )
