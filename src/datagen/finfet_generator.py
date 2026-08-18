import numpy as np

class FinFETGenerator:
    """
    Generates a synthetic FinFET-style array structure.
    Parallel vertical fin lines, crossed by horizontal gate bars.
    """
    def __init__(self, fin_pitch: int, gate_pitch: int, num_gates: int, defect_rate: float, irregularity: bool):
        self.fin_pitch = fin_pitch
        self.gate_pitch = gate_pitch
        self.num_gates = num_gates
        self.defect_rate = defect_rate
        self.irregularity = irregularity
        
    def generate(self, width: int, height: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = np.zeros((height, width), dtype=np.float32)
        
        fin_width = max(1, self.fin_pitch // 3)
        gate_width = max(2, self.gate_pitch // 2)
        
        # Fins (vertical)
        for x in range(0, width, self.fin_pitch):
            img[:, x:min(x + fin_width, width)] = 0.6
            
        # Gates (horizontal)
        for y in range(0, height, self.gate_pitch):
            img[y:min(y + gate_width, height), :] = np.maximum(img[y:min(y + gate_width, height), :], 0.8)
            
            # Simulated gate crossings (distinctive structures)
            if self.num_gates > 1:
                y2 = y + gate_width + 2
                img[y2:min(y2 + gate_width // 2, height), :] = np.maximum(
                    img[y2:min(y2 + gate_width // 2, height), :], 0.7
                )
                
            # Defects (e.g. missing sections of gate)
            if rng.random() < self.defect_rate:
                x_def = rng.integers(0, width - self.fin_pitch)
                img[y:min(y + gate_width, height), x_def:x_def+self.fin_pitch] = 0.0
                
        return img
