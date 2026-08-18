import numpy as np

class DRAMGenerator:
    """
    Generates a synthetic DRAM-style array structure.
    Based on 6F^2 or 8F^2 cell layouts with periodic word lines and bit lines.
    """
    def __init__(self, f_size: int, unit_cell: str, defect_rate: float, irregularity: bool):
        self.f = f_size
        self.unit_cell = unit_cell # "6F2" or "8F2"
        self.defect_rate = defect_rate
        self.irregularity = irregularity
        
    def generate(self, width: int, height: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = np.zeros((height, width), dtype=np.float32)
        
        # Word lines (horizontal)
        wl_pitch = 2 * self.f if self.unit_cell == "6F2" else 3 * self.f
        wl_width = self.f
        
        # Bit lines (vertical)
        bl_pitch = 3 * self.f if self.unit_cell == "6F2" else 2 * self.f
        bl_width = self.f
        
        for y in range(0, height, wl_pitch):
            img[y:min(y + wl_width, height), :] = 0.4
            
        for x in range(0, width, bl_pitch):
            img[:, x:min(x + bl_width, width)] = np.maximum(img[:, x:min(x + bl_width, width)], 0.5)
            
        # Contacts at intersections
        for y in range(0, height, wl_pitch):
            for x in range(0, width, bl_pitch):
                # Defect rate: skip some contacts
                if rng.random() > self.defect_rate:
                    cy = y + wl_width // 2
                    cx = x + bl_width // 2
                    
                    if self.irregularity:
                        cy += rng.integers(-1, 2)
                        cx += rng.integers(-1, 2)
                        
                    rad = self.f // 2
                    y0, y1 = max(0, cy - rad), min(height, cy + rad + 1)
                    x0, x1 = max(0, cx - rad), min(width, cx + rad + 1)
                    
                    if y1 > y0 and x1 > x0:
                        img[y0:y1, x0:x1] = 0.9 # High intensity contact
                        
        return img
