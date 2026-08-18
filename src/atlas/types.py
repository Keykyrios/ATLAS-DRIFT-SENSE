"""Core data types for the ATLAS pipeline.

Defines the structured containers passed between pipeline stages:
Candidate, PoseEstimate, ATLASResult, and InformativenessReport.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class Candidate:
    x: float
    y: float
    scale: float
    score: float
    rotation: float = 0.0
    d_topo: float = 1.0
    d_topo_norm: float = 1.0
    
@dataclass
class PoseEstimate:
    scale: float
    theta: float
    tx: float
    ty: float

@dataclass
class ATLASResult:
    x: float
    y: float
    confidence: float
    is_low_informativeness: bool
    runtime_ms: float
    stage_timings: Dict[str, float]
    tie_break_applied: bool = False
    
@dataclass
class InformativenessReport:
    score: float
    tau_star_x: int
    tau_star_y: int
    flag_low_confidence: bool
