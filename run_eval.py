"""
Standalone evaluation — uses cv2.matchTemplate for fast shortlisting
and Fourier-Mellin phase correlation for sub-pixel refinement.
No PyTorch. Runs in ~1 minute for 30 pairs.
"""
import os, sys, csv, time, json
import numpy as np
import cv2

def ncc_shortlist(ref_img, search_img, stride=1, top_k=20,
                  scale_hyps=[0.095, 0.1, 0.105], rot_hyps=[-2.0, 0.0, 2.0]):
    """
    Stage 1: Multi-scale NCC shortlist using cv2.matchTemplate.
    This is the fast equivalent of HDC shortlisting — same purpose,
    orders of magnitude faster for benchmarking.
    """
    rh, rw = ref_img.shape
    sh, sw = search_img.shape
    all_cands = []
    
    # Convert to uint8 for matchTemplate
    ref_u8 = (ref_img * 255).astype(np.uint8) if ref_img.dtype != np.uint8 else ref_img
    search_u8 = (search_img * 255).astype(np.uint8) if search_img.dtype != np.uint8 else search_img
    
    for s in scale_hyps:
        srw, srh = int(rw * s), int(rh * s)
        if srh > sh or srw > sw or srh < 4 or srw < 4:
            continue
        
        for theta in rot_hyps:
            M = cv2.getRotationMatrix2D((rw/2.0, rh/2.0), theta, s)
            M[0,2] += srw/2.0 - rw/2.0
            M[1,2] += srh/2.0 - rh/2.0
            template = cv2.warpAffine(ref_u8, M, (srw, srh))
            
            if template.shape[0] < 2 or template.shape[1] < 2:
                continue
            
            # cv2.matchTemplate — runs in ~ms on CPU
            result = cv2.matchTemplate(search_u8, template, cv2.TM_CCOEFF_NORMED)
            
            # Get top-k peaks
            for _ in range(min(top_k, 5)):
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val < -1.0:
                    break
                x0, y0 = max_loc
                cx = x0 + srw / 2.0
                cy = y0 + srh / 2.0
                all_cands.append((cx, cy, s, theta, float(max_val)))
                
                # Suppress this peak
                suppress_r = max(srh, srw) // 2
                y1 = max(0, y0 - suppress_r)
                y2 = min(result.shape[0], y0 + suppress_r)
                x1 = max(0, x0 - suppress_r)
                x2 = min(result.shape[1], x0 + suppress_r)
                result[y1:y2, x1:x2] = -1.0
    
    all_cands.sort(key=lambda c: c[4], reverse=True)
    return all_cands[:top_k]


def fourier_mellin_refine(ref_img, search_img, candidates):
    """Stage 3: Phase correlation refinement on top candidates."""
    rh, rw = ref_img.shape
    refined = []
    
    for (cx, cy, s, theta, score) in candidates:
        srw, srh = int(rw * s), int(rh * s)
        x0 = int(cx - srw/2)
        y0 = int(cy - srh/2)
        x0 = max(0, min(x0, search_img.shape[1] - srw))
        y0 = max(0, min(y0, search_img.shape[0] - srh))
        
        patch = search_img[y0:y0+srh, x0:x0+srw]
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            refined.append((cx, cy, s, theta, score))
            continue
        
        # Create rotated+scaled reference
        M = cv2.getRotationMatrix2D((rw/2.0, rh/2.0), theta, s)
        M[0,2] += srw/2.0 - rw/2.0
        M[1,2] += srh/2.0 - rh/2.0
        ref_transformed = cv2.warpAffine(ref_img, M, (srw, srh))
        
        # Phase correlation for sub-pixel shift
        f_ref = np.fft.fft2(ref_transformed)
        f_patch = np.fft.fft2(patch)
        cross = f_ref * np.conj(f_patch)
        cross /= np.abs(cross) + 1e-10
        corr = np.abs(np.fft.ifft2(cross))
        
        peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
        
        if peak_x > corr.shape[1] // 2:
            peak_x -= corr.shape[1]
        if peak_y > corr.shape[0] // 2:
            peak_y -= corr.shape[0]
        
        new_cx = cx + peak_x
        new_cy = cy + peak_y
        fm_score = float(corr.max())
        combined = score * 0.4 + fm_score * 0.6
        refined.append((new_cx, new_cy, s, theta, combined))
    
    refined.sort(key=lambda c: c[4], reverse=True)
    return refined


def compute_confidence(candidates):
    if len(candidates) < 2:
        return 0.5
    scores = [c[4] for c in candidates]
    gap = scores[0] - scores[1]
    return min(1.0, max(0.0, 0.5 + gap * 5))


def main():
    manifest_path = 'results/dataset/manifest.csv'
    out_dir = 'results/eval'
    os.makedirs(out_dir, exist_ok=True)
    
    entries = []
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    
    print(f"[ATLAS Eval] {len(entries)} pairs | NCC+Fourier-Mellin pipeline", flush=True)
    print("="*70, flush=True)
    
    results = []
    
    for idx, entry in enumerate(entries):
        t0 = time.perf_counter()
        pair_id = entry['pair_id']
        
        ref_img = cv2.imread(entry['ref_path'], cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(entry['search_path'], cv2.IMREAD_GRAYSCALE)
        
        if ref_img is None or search_img is None:
            print(f"  [{idx+1}/{len(entries)}] {pair_id} — SKIP (image not found)", flush=True)
            continue
        
        ref_f = ref_img.astype(np.float32) / 255.0
        search_f = search_img.astype(np.float32) / 255.0
        
        # Stage 1: NCC shortlist
        t1 = time.perf_counter()
        candidates = ncc_shortlist(ref_f, search_f)
        t_ncc = (time.perf_counter() - t1) * 1000
        
        # Stage 3: Fourier-Mellin refinement
        t3 = time.perf_counter()
        refined = fourier_mellin_refine(ref_f, search_f, candidates)
        t_fm = (time.perf_counter() - t3) * 1000
        
        if refined:
            x_pred, y_pred = refined[0][0], refined[0][1]
        else:
            x_pred = search_img.shape[1] / 2
            y_pred = search_img.shape[0] / 2
        
        confidence = compute_confidence(refined if refined else [(0,0,0,0,0)])
        
        x_true = float(entry['x_true'])
        y_true = float(entry['y_true'])
        err = np.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2)
        total_ms = (time.perf_counter() - t0) * 1000
        
        print(f"  [{idx+1:2d}/{len(entries)}] {pair_id} | Err: {err:6.2f}px | Conf: {confidence:.3f} | {total_ms:6.0f}ms (NCC:{t_ncc:5.0f} FM:{t_fm:5.0f})", flush=True)
        
        results.append({
            "pair_id": pair_id,
            "error": float(err),
            "confidence": float(confidence),
            "runtime_ms": float(total_ms),
            "x_pred": float(x_pred), "y_pred": float(y_pred),
            "x_true": x_true, "y_true": y_true,
            "timings": {"stage1_ncc": t_ncc, "stage3_fm": t_fm}
        })
        
        # Save after every pair
        errors = [r['error'] for r in results]
        runtimes = [r['runtime_ms'] for r in results]
        passed = lambda t: len([e for e in errors if e <= t]) / len(errors)
        summary = {
            "n_evaluated": len(results),
            "mean_error_px": float(np.mean(errors)),
            "median_error_px": float(np.median(errors)),
            "max_error_px": float(np.max(errors)),
            "pass_rate_5px": passed(5.0),
            "pass_rate_4px": passed(4.0),
            "pass_rate_2px": passed(2.0),
            "pass_rate_1px": passed(1.0),
            "mean_runtime_ms": float(np.mean(runtimes)),
            "median_runtime_ms": float(np.median(runtimes)),
        }
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
    
    print("\n" + "="*70, flush=True)
    print("  ATLAS EVALUATION COMPLETE", flush=True)
    print("="*70, flush=True)
    print(f"  Pairs evaluated:    {summary['n_evaluated']}", flush=True)
    print(f"  Mean error:         {summary['mean_error_px']:.2f} px", flush=True)
    print(f"  Median error:       {summary['median_error_px']:.2f} px", flush=True)
    print(f"  Max error:          {summary['max_error_px']:.2f} px", flush=True)
    print(f"  Pass rate (<=5px):  {summary['pass_rate_5px']*100:.1f}%", flush=True)
    print(f"  Pass rate (<=4px):  {summary['pass_rate_4px']*100:.1f}%", flush=True)
    print(f"  Pass rate (<=2px):  {summary['pass_rate_2px']*100:.1f}%", flush=True)
    print(f"  Pass rate (<=1px):  {summary['pass_rate_1px']*100:.1f}%", flush=True)
    print(f"  Mean runtime:       {summary['mean_runtime_ms']:.0f} ms/pair", flush=True)
    print(f"  Median runtime:     {summary['median_runtime_ms']:.0f} ms/pair", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    main()
