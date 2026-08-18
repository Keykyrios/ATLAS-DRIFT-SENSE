import csv
import os
from typing import Dict, Any, List

class ManifestManager:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.headers = [
            "pair_id", "architecture", "ref_path", "search_path", 
            "seed_ref", "seed_search", "noise_a_ref", "noise_a_search", 
            "noise_b", "psf_sigma", "scale", "rotation", 
            "x_true", "y_true"
        ]
        
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                
    def add_entry(self, data: Dict[str, Any]):
        with open(self.manifest_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [data.get(h, "") for h in self.headers]
            writer.writerow(row)
            
    def get_all(self) -> List[Dict[str, Any]]:
        results = []
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
        return results
