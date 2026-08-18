import argparse
import json
import cv2
import sys
from src.atlas.pipeline import ATLASPipeline

def main():
    parser = argparse.ArgumentParser(description="ATLAS Navigation-Error Recovery CLI")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    parser.add_argument("--search", type=str, required=True, help="Path to search image")
    parser.add_argument("--rgb", action="store_true", help="Enable RGB processing (not fully implemented in CLI yet)")
    parser.add_argument("--json", action="store_true", help="Output full JSON instead of just x,y")
    args = parser.parse_args()

    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    
    if ref_img is None or search_img is None:
        print("Error: Could not read one or both images.", file=sys.stderr)
        sys.exit(1)

    pipeline = ATLASPipeline(config_path="configs/default.yaml")
    
    result = pipeline.process(ref_img, search_img)
    
    if args.json:
        output = {
            "x": result.x,
            "y": result.y,
            "confidence": result.confidence,
            "is_low_informativeness": result.is_low_informativeness,
            "tie_break_applied": result.tie_break_applied,
            "runtime_ms": result.runtime_ms,
            "stage_timings": result.stage_timings
        }
        print(json.dumps(output, indent=2))
    else:
        # The expected output format per the problem statement
        print(f"{result.x},{result.y}")

if __name__ == "__main__":
    main()
