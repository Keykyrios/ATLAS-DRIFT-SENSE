import os
import cv2
import json
from src.datagen.manifest import ManifestManager
from src.atlas.pipeline import ATLASPipeline
from .metrics import euclidean_error, pass_rate_at_threshold

class EvaluationRunner:
    def __init__(self, manifest_path: str, out_dir: str):
        self.manifest = ManifestManager(manifest_path)
        self.out_dir = out_dir
        self.pipeline = ATLASPipeline()
        
    def run_all(self):
        entries = self.manifest.get_all()
        results = []
        
        for idx, entry in enumerate(entries):
            print(f"Evaluating {idx+1}/{len(entries)}: {entry['pair_id']}")
            
            ref_path = entry['ref_path']
            search_path = entry['search_path']
            
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
            
            if ref_img is None or search_img is None:
                continue
                
            res = self.pipeline.process(ref_img, search_img)
            
            x_true = float(entry['x_true'])
            y_true = float(entry['y_true'])
            
            err = euclidean_error(res.x, res.y, x_true, y_true)
            
            results.append({
                "pair_id": entry['pair_id'],
                "error": err,
                "confidence": res.confidence,
                "is_low_info": res.is_low_informativeness,
                "runtime_ms": res.runtime_ms,
                "x_pred": res.x,
                "y_pred": res.y,
                "x_true": x_true,
                "y_true": y_true,
                "timings": res.stage_timings
            })
            
        # Compute summary
        errors = [r['error'] for r in results]
        summary = {
            "mean_error": float(np.mean(errors)) if errors else 0.0,
            "median_error": float(np.median(errors)) if errors else 0.0,
            "max_error": float(np.max(errors)) if errors else 0.0,
            "pass_rate_5px": pass_rate_at_threshold(errors, 5.0),
            "pass_rate_4px": pass_rate_at_threshold(errors, 4.0),
            "pass_rate_2px": pass_rate_at_threshold(errors, 2.0),
            "pass_rate_1px": pass_rate_at_threshold(errors, 1.0)
        }
        
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "metrics.json"), 'w') as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
            
        print(f"Evaluation complete. Mean error: {summary['mean_error']:.2f}px")
        return summary
