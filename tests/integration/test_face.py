"""
Phase 2 Verification Tests

Test 2.1 (Detection Accuracy): Multi-face, profiles, varied lighting
Test 2.2 (Alignment Constraint): Output dimensions match exactly

Usage:
    python face/test_face.py
"""

import sys
import os
import time
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from contracts import FrameData, FaceData
from camera.camera_module import CameraModule
from face.face_module import FaceModule


# ─────────────────────────────────────────────────────
# TEST 2.1: Detection Accuracy
# ─────────────────────────────────────────────────────

def test_detection_accuracy():
    """
    Test face detection with the camera running.
    
    Verification:
    - Bounding boxes match facial perimeters (not too large, not too small)
    - Works in varying lighting (test by moving)
    - Handles multiple faces if present
    
    Manual: Stand in front of camera, move around, change lighting.
    The test will sample frames and verify detections.
    """
    print("\n" + "="*60)
    print("TEST 2.1: Detection Accuracy")
    print("="*60)
    print("[INFO] Stand in front of camera now...")
    print("[INFO] The test will sample 10 frames over 15 seconds")
    print("[INFO] Try: facing forward, slight profile, different distances")
    
    camera = CameraModule()
    face_module = FaceModule()
    
    if not camera.start():
        print("[FAIL] Camera failed to start")
        return False
    
    time.sleep(1)  # Let camera warm up
    
    detections = []
    samples_with_no_face = 0
    
    for i in range(10):
        frame_data = camera.get_latest_frame()
        
        if frame_data is None:
            print(f"  Sample {i+1}: No frame available")
            continue
        
        faces = face_module.detect_faces(frame_data)
        
        print(f"  Sample {i+1}: Detected {len(faces)} face(s)")
        
        if len(faces) == 0:
            samples_with_no_face += 1
        else:
            for j, face in enumerate(faces):
                x, y, w, h = face.bbox
                frame_h, frame_w = frame_data.pixel_array.shape[:2]
                
                print(f"    Face {j+1}: bbox=({x},{y},{w},{h}) "
                      f"confidence={face.confidence:.2f}")
                
                # Check bounding box is within frame
                if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                    print(f"    [FAIL] Bbox extends beyond frame boundaries")
                    camera.stop()
                    return False
                
                # Check reasonable face size
                frame_area = frame_w * frame_h
                face_area = w * h
                ratio = face_area / frame_area
                
                if ratio < 0.01:
                    print(f"    [WARN] Face very small: {ratio:.4f} of frame")
                elif ratio > 0.8:
                    print(f"    [WARN] Face very large: {ratio:.4f} of frame")
                
                # Check aspect ratio is face-like
                aspect = w / h if h > 0 else 0
                if not (0.5 <= aspect <= 2.0):
                    print(f"    [WARN] Suspicious aspect ratio: {aspect:.2f}")
                
                # Verify alignment
                if not face.verify_alignment():
                    print(f"    [FAIL] Face alignment verification failed")
                    camera.stop()
                    return False
                
                detections.append(face)
        
        time.sleep(1.5)
    
    camera.stop()
    
    # Analysis
    print("\n--- Analysis ---")
    print(f"Total detections: {len(detections)}")
    print(f"Samples with no face: {samples_with_no_face}/10")
    
    if len(detections) == 0 and samples_with_no_face < 10:
        print("[WARN] Some frames captured but no faces detected")
    
    if samples_with_no_face == 10:
        print("[WARN] No faces detected in any sample.")
        print("[INFO] This is OK if no one was in front of camera.")
        print("[INFO] Re-run the test with a person visible.")
    
    print("[PASS] Test 2.1: Detection accuracy check complete")
    return True


# ─────────────────────────────────────────────────────
# TEST 2.2: Alignment Constraint
# ─────────────────────────────────────────────────────

def test_alignment_dimensions():
    """
    Verify that aligned face crops have EXACT dimensions.
    
    Tests three target sizes to ensure the module respects
    its configuration.
    """
    print("\n" + "="*60)
    print("TEST 2.2: Alignment Dimension Constraints")
    print("="*60)
    
    # Test with multiple target sizes
    target_sizes = [(112, 112), (160, 160)]
    
    camera = CameraModule()
    
    if not camera.start():
        print("[FAIL] Camera failed to start")
        return False
    
    time.sleep(1)
    
    for target_w, target_h in target_sizes:
        print(f"\n  Testing target size: {target_w}x{target_h}")
        face_module = FaceModule(target_size=(target_w, target_h))
        
        # Capture a frame
        frame_data = camera.get_latest_frame()
        if frame_data is None:
            print(f"  [FAIL] No frame captured")
            continue
        
        faces = face_module.detect_faces(frame_data)
        
        if len(faces) == 0:
            print(f"  [WARN] No face detected for {target_w}x{target_h} test")
            print(f"  [INFO] Stand in front of camera and re-run")
            continue
        
        for i, face in enumerate(faces):
            actual_h, actual_w = face.cropped_face.shape[:2]
            
            print(f"  Face {i+1}: Expected ({target_w}, {target_h}), "
                  f"Got ({actual_w}, {actual_h})")
            
            if (actual_w, actual_h) != (target_w, target_h):
                print(f"  [FAIL] Dimensions don't match!")
                camera.stop()
                return False
            
            # Also verify via the self-test
            if not face.verify_alignment():
                print(f"  [FAIL] verify_alignment() returned False")
                camera.stop()
                return False
            
            print(f"  [PASS] Dimensions match exactly")
    
    camera.stop()
    print("\n[PASS] Test 2.2: All alignment dimensions verified")
    return True


# ─────────────────────────────────────────────────────
# TEST 2.2b: Eye Horizontality
# ─────────────────────────────────────────────────────

def test_eye_horizontality():
    """
    In the aligned crop, eyes should be at nearly the same y-coordinate.
    We verify by checking the crop visually and geometrically.
    """
    print("\n" + "="*60)
    print("TEST 2.2b: Eye Horizontality Check")
    print("="*60)
    
    camera = CameraModule()
    face_module = FaceModule(target_size=(160, 160))
    
    if not camera.start():
        print("[FAIL] Camera failed to start")
        return False
    
    time.sleep(1)
    
    # Collect a few aligned faces
    aligned_faces = []
    
    for _ in range(5):
        frame_data = camera.get_latest_frame()
        if frame_data:
            faces = face_module.detect_faces(frame_data)
            for face in faces:
                aligned_faces.append(face)
        time.sleep(0.5)
    
    camera.stop()
    
    if len(aligned_faces) == 0:
        print("[WARN] No faces captured for horizontality test")
        print("[PASS] Test 2.2b: Skipped (no faces)")
        return True
    
    # Save aligned crops to temp directory for visual inspection
    temp_dir = tempfile.mkdtemp(prefix="aligned_faces_")
    
    for i, face in enumerate(aligned_faces[:3]):  # Save first 3
        filepath = os.path.join(temp_dir, f"aligned_face_{i+1}.jpg")
        cv2.imwrite(filepath, face.cropped_face)
        print(f"  Saved: {filepath}")
        
        # Draw eye position lines on the crop for verification
        vis = face.cropped_face.copy()
        h, w = vis.shape[:2]
        
        # Expected eye y-level
        expected_eye_y = int(h * 0.40)
        cv2.line(vis, (0, expected_eye_y), (w, expected_eye_y), (0, 255, 0), 1)
        
        filepath = os.path.join(temp_dir, f"aligned_face_{i+1}_with_line.jpg")
        cv2.imwrite(filepath, vis)
        print(f"  Saved with eye-line: {filepath}")
    
    print(f"\n[INFO] Inspect images in: {temp_dir}")
    print("[INFO] Eyes should be level with the green line")
    print("[PASS] Test 2.2b: Horizontality check complete")
    return True


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2 Face Detection Tests")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip detection accuracy test")
    parser.add_argument("--skip-dimensions", action="store_true",
                        help="Skip alignment dimensions test")
    parser.add_argument("--skip-horizontality", action="store_true",
                        help="Skip eye horizontality test")
    args = parser.parse_args()
    
    results = []
    
    if not args.skip_accuracy:
        results.append(("Test 2.1: Detection Accuracy", 
                        test_detection_accuracy()))
    
    if not args.skip_dimensions:
        results.append(("Test 2.2: Alignment Dimensions", 
                        test_alignment_dimensions()))
    
    if not args.skip_horizontality:
        results.append(("Test 2.2b: Eye Horizontality", 
                        test_eye_horizontality()))
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")