import sys
import os
import time
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import psutil

from contracts import FrameData
from camera.camera_module import CameraModule


# ─────────────────────────────────────────────────────
# TEST 1.1: Hardware Loop (Memory Stability)
# ─────────────────────────────────────────────────────

def test_memory_stability(duration_seconds: int = 600):
    """
    Run camera continuously for `duration_seconds`.
    Verify:
    - Frame queue stays stable (always 1 frame in buffer)
    - Memory usage does NOT scale linearly over time
    - FPS remains consistent
    """
    print("\n" + "="*60)
    print("TEST 1.1: Memory Stability (10-minute run)")
    print("="*60)
    
    camera = CameraModule()
    
    if not camera.start():
        print("[FAIL] Camera failed to start")
        return False
    
    process = psutil.Process(os.getpid())
    
    # Take baseline memory
    initial_memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Initial memory: {initial_memory_mb:.2f} MB")
    
    # Sample memory every 30 seconds
    memory_samples = []
    frame_samples = []
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_seconds:
            frame = camera.get_latest_frame()
            
            # Verify frame is valid every time
            if frame is not None and not frame.verify_integrity():
                print(f"[FAIL] Frame #{frame.frame_number} failed integrity check")
                camera.stop()
                return False
            
            # Sample stats every 30 seconds
            elapsed = time.time() - start_time
            if len(memory_samples) == 0 or elapsed - memory_samples[-1][0] >= 30:
                current_memory_mb = process.memory_info().rss / (1024 * 1024)
                stats = camera.get_stats()
                memory_samples.append((elapsed, current_memory_mb))
                frame_samples.append((elapsed, stats['frames_captured']))
                
                print(f"  [{elapsed:.0f}s] Memory: {current_memory_mb:.2f} MB | "
                      f"Frames: {stats['frames_captured']} | FPS: {stats['fps']}")
            
            time.sleep(0.01)  # Don't hammer the CPU
    
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
    
    finally:
        camera.stop()
    
    final_memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # ── Analysis ──
    print("\n--- Analysis ---")
    print(f"Initial memory: {initial_memory_mb:.2f} MB")
    print(f"Final memory:   {final_memory_mb:.2f} MB")
    print(f"Memory change:  {final_memory_mb - initial_memory_mb:+.2f} MB")
    
    # Check: memory should not grow more than 100MB (allowing some overhead)
    memory_growth = final_memory_mb - initial_memory_mb
    if memory_growth > 100:
        print(f"[FAIL] Memory grew by {memory_growth:.2f} MB — possible leak detected")
        return False
    
    # Check: frame rate should be reasonable
    total_frames = camera._frame_count
    total_time = time.time() - start_time
    avg_fps = total_frames / total_time if total_time > 0 else 0
    print(f"Average FPS: {avg_fps:.2f}")
    
    if avg_fps < 5:
        print(f"[FAIL] FPS too low: {avg_fps:.2f}")
        return False
    
    # Check: buffer depth is always 1
    stats = camera.get_stats()
    if stats['buffer_depth'] != 1:
        print(f"[FAIL] Buffer depth is {stats['buffer_depth']}, should be 1")
        return False
    
    print("[PASS] Test 1.1: Memory stability verified")
    return True


# ─────────────────────────────────────────────────────
# TEST 1.2: Frame Integrity (Dump every 50th frame)
# ─────────────────────────────────────────────────────

def test_frame_integrity(output_dir: str = None):
    """
    Run camera and save every 50th frame as .jpg.
    Verify each saved frame is crisp and free of artifacts.
    
    Manual step: Open the output directory and inspect images.
    """
    print("\n" + "="*60)
    print("TEST 1.2: Frame Integrity (every 50th frame dump)")
    print("="*60)
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="frame_test_")
    
    print(f"Output directory: {output_dir}")
    
    camera = CameraModule()
    
    if not camera.start():
        print("[FAIL] Camera failed to start")
        return False
    
    saved_frames = []
    required_frames = 5  # Save 5 frames total (every 50th => need ~250 frames)
    last_saved_number = -1
    
    try:
        while len(saved_frames) < required_frames:
            frame_data = camera.get_latest_frame()
            
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            # Save every 50th frame
            if (frame_data.frame_number % 50 == 0 and 
                frame_data.frame_number != last_saved_number):
                
                filename = f"frame_{frame_data.frame_number:06d}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                cv2.imwrite(filepath, frame_data.pixel_array)
                
                # Verify the saved file
                file_size_kb = os.path.getsize(filepath) / 1024
                
                print(f"  Saved {filename} | "
                      f"Size: {file_size_kb:.1f} KB | "
                      f"Resolution: {frame_data.resolution}")
                
                # Check: file should be at least 5KB (not a blank image)
                if file_size_kb < 5:
                    print(f"  [WARN] {filename} is suspiciously small ({file_size_kb:.1f} KB)")
                
                saved_frames.append(filepath)
                last_saved_number = frame_data.frame_number
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
    
    finally:
        camera.stop()
    
    # ── Analysis ──
    print("\n--- Analysis ---")
    print(f"Saved {len(saved_frames)} frames to: {output_dir}")
    
    for filepath in saved_frames:
        # Read back and check
        img = cv2.imread(filepath)
        if img is None:
            print(f"[FAIL] Cannot read back {filepath}")
            return False
        
        # Check: not completely black
        mean_pixel = np.mean(img)
        if mean_pixel < 5:
            print(f"[WARN] {os.path.basename(filepath)} is very dark (mean={mean_pixel:.1f})")
        
        # Check: not completely uniform (some variance)
        std_pixel = np.std(img)
        if std_pixel < 5:
            print(f"[WARN] {os.path.basename(filepath)} has very low variance "
                  f"(std={std_pixel:.1f}) — might be blank")
        
        print(f"  {os.path.basename(filepath)}: mean={mean_pixel:.1f}, std={std_pixel:.1f}")
    
    print(f"\n[INFO] Manual step: Inspect images in {output_dir}")
    print("[PASS] Test 1.2: Frame integrity check complete")
    return True


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1 Camera Module Tests")
    parser.add_argument("--quick", action="store_true", 
                        help="Run memory test for only 60 seconds instead of 600")
    parser.add_argument("--skip-memory", action="store_true",
                        help="Skip the memory stability test")
    parser.add_argument("--skip-integrity", action="store_true",
                        help="Skip the frame integrity test")
    args = parser.parse_args()
    
    results = []
    
    if not args.skip_memory:
        duration = 60 if args.quick else 600
        results.append(("Test 1.1: Memory Stability", 
                        test_memory_stability(duration)))
    
    if not args.skip_integrity:
        results.append(("Test 1.2: Frame Integrity", 
                        test_frame_integrity()))
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")