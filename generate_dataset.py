import argparse
import os
import yaml
import cv2
import numpy as np
from uuid import uuid4

from src.datagen.dram_generator import DRAMGenerator
from src.datagen.finfet_generator import FinFETGenerator
from src.datagen.noise_models import apply_mixed_poisson_gaussian_noise
from src.datagen.degradations import apply_psf_blur, apply_edge_brightening
from src.datagen.transform import crop_reference
from src.datagen.manifest import ManifestManager

def generate_pair(config: dict, out_dir: str, pair_id: str, is_dram: bool):
    rng = np.random.default_rng()
    
    # Generate large base image (Search Image base)
    seed_base = rng.integers(0, 1000000)
    
    if is_dram:
        generator = DRAMGenerator(
            f_size=config.get("feature_size_f", 10),
            unit_cell=config.get("unit_cell_type", "6F2"),
            defect_rate=config.get("defect_rate", 0.005),
            irregularity=config.get("boundary_irregularity", True)
        )
    else:
        generator = FinFETGenerator(
            fin_pitch=config.get("fin_pitch", 12),
            gate_pitch=config.get("gate_pitch", 20),
            num_gates=config.get("num_gates", 2),
            defect_rate=config.get("defect_rate", 0.002),
            irregularity=config.get("boundary_irregularity", True)
        )
        
    search_base = generator.generate(1000, 1000, seed_base)
    
    # 2. Extract Reference Image true patch
    # Target scale between 0.09 and 0.11
    scale_ratio = rng.uniform(0.09, 0.11)
    rotation_deg = rng.uniform(-2.0, 2.0)
    
    # Pick a random center for the reference in the search image
    # A 1000x1000 reference at scale 0.1 corresponds to a 100x100 patch in search.
    # So margin in search image is 100 / 2 = 50. We use 60 for safety with rotation.
    margin = 60
    center_x = rng.integers(margin, 1000 - margin)
    center_y = rng.integers(margin, 1000 - margin)
    
    ref_base, transform_matrix = crop_reference(
        search_base, scale_ratio, rotation_deg, center_x, center_y, ref_size=1000
    )
    
    # 3. Apply independent noise and degradations
    # Search image should be slightly noisier (higher 'a')
    noise_cfg = config.get("noise", {})
    a_min, a_max = noise_cfg.get("a_range", [0.5, 1.5])
    b_min, b_max = noise_cfg.get("b_range", [2.0, 5.0])
    psf_sigma = rng.uniform(*noise_cfg.get("psf_sigma_range", [0.5, 1.2]))
    boost = noise_cfg.get("edge_brightening_boost", 1.2)
    
    a_ref = rng.uniform(a_min, (a_max + a_min) / 2)
    a_search = rng.uniform((a_max + a_min) / 2, a_max) # Harder noise
    b = rng.uniform(b_min, b_max)
    
    seed_ref = rng.integers(0, 1000000)
    seed_search = rng.integers(0, 1000000)
    
    # Apply to Ref
    ref_noisy = apply_mixed_poisson_gaussian_noise(ref_base, a_ref, b, seed_ref)
    ref_blur = apply_psf_blur(ref_noisy, psf_sigma)
    ref_final = apply_edge_brightening(ref_blur, boost)
    
    # Apply to Search
    search_noisy = apply_mixed_poisson_gaussian_noise(search_base, a_search, b, seed_search)
    search_blur = apply_psf_blur(search_noisy, psf_sigma)
    search_final = apply_edge_brightening(search_blur, boost)
    
    # 4. Save to disk
    ref_path = os.path.join(out_dir, f"{pair_id}_ref.png")
    search_path = os.path.join(out_dir, f"{pair_id}_search.png")
    
    cv2.imwrite(ref_path, (ref_final * 255).astype(np.uint8))
    cv2.imwrite(search_path, (search_final * 255).astype(np.uint8))
    
    return {
        "pair_id": pair_id,
        "architecture": "DRAM" if is_dram else "FinFET",
        "ref_path": ref_path,
        "search_path": search_path,
        "seed_ref": seed_ref,
        "seed_search": seed_search,
        "noise_a_ref": a_ref,
        "noise_a_search": a_search,
        "noise_b": b,
        "psf_sigma": psf_sigma,
        "scale": scale_ratio,
        "rotation": rotation_deg,
        "x_true": center_x,
        "y_true": center_y
    }

def main():
    parser = argparse.ArgumentParser(description="ATLAS Synthetic Dataset Generator")
    parser.add_argument("--arch", choices=["dram", "finfet", "both"], default="both", help="Architecture to generate")
    parser.add_argument("--num-pairs", type=int, default=30, help="Number of pairs to generate")
    parser.add_argument("--out-dir", type=str, default="results/dataset", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    manifest = ManifestManager(os.path.join(args.out_dir, "manifest.csv"))
    
    with open("configs/dram.yaml", 'r') as f:
        dram_cfg = yaml.safe_load(f)
    with open("configs/finfet.yaml", 'r') as f:
        finfet_cfg = yaml.safe_load(f)
        
    for i in range(args.num_pairs):
        pair_id = str(uuid4())[:8]
        
        if args.arch == "both":
            is_dram = (i % 2 == 0)
        else:
            is_dram = (args.arch == "dram")
            
        cfg = dram_cfg if is_dram else finfet_cfg
        
        print(f"Generating pair {i+1}/{args.num_pairs} ({'DRAM' if is_dram else 'FinFET'}) - {pair_id}")
        data = generate_pair(cfg, args.out_dir, pair_id, is_dram)
        manifest.add_entry(data)
        
    print(f"Dataset generation complete. Manifest saved to {args.out_dir}/manifest.csv")

if __name__ == "__main__":
    main()
