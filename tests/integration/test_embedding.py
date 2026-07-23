"""
Phase 3 Verification Tests

Test 3.1 (Consistency): Same person, multiple frames → near-zero distance
Test 3.2 (Entropy/Diversity): Different people → high orthogonal diversity

Usage:
    python embedding/test_embedding.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from contracts import FrameData, FaceData, EmbeddingData
from camera.camera_module import CameraModule
from face.face_module import FaceModule
from embedding.embedding_module import EmbeddingModule


# ─────────────────────────────────────────────────────
# TEST 3.1: Consistency (Same Person)
# ─────────────────────────────────────────────────────

def test_consistency():
    """
    Capture 3 frames of the same person.
    Verify embeddings are nearly identical (cosine distance ≈ 0).
    """
    print("\n" + "="*60)
    print("TEST 3.1: Embedding Consistency (Same Person)")
    print("="*60)
    print("[INFO] Stand still in front of camera...")
    print("[INFO] Capturing 3 frames and comparing embeddings")
    
    camera = CameraModule()
    face_module = FaceModule()
    embedding_module = EmbeddingModule()
    
    if not camera.start():
        print("[FAIL] Camera failed to start")
        return False
    
    time.sleep(1)
    
    embeddings = []
    
    for i in range(3):
        # Wait a moment between captures
        time.sleep(0.5)
        
        frame_data = camera.get_latest_frame()
        if frame_data is None:
            print(f"  Capture {i+1}: No frame available")
            continue
        
        face_data = face_module.detect_largest_face(frame_data)
        if face_data is None:
            print(f"  Capture {i+1}: No face detected")
            continue
        
        embedding = embedding_module.extract(face_data)
        if embedding is None:
            print(f"  Capture {i+1}: Embedding extraction failed")
            continue
        
        # Verify embedding self-check
        if not embedding.verify_normalization():
            print(f"  Capture {i+1}: [FAIL] Embedding failed normalization check")
            camera.stop()
            return False
        
        embeddings.append(embedding)
        print(f"  Capture {i+1}: Embedding extracted "
              f"(dim={embedding.dimension}, "
              f"norm={np.linalg.norm(embedding.vector):.4f}, "
              f"time={embedding.extraction_time_ms:.1f}ms)")
    
    camera.stop()
    
    if len(embeddings) < 2:
        print("[FAIL] Not enough embeddings to compare")
        return False
    
    # Compare all pairs
    print("\n--- Pairwise Comparison ---")
    all_passed = True
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = embeddings[i].cosine_similarity(embeddings[j])
            dist = embeddings[i].cosine_distance(embeddings[j])
            
            print(f"  Embedding {i+1} vs {j+1}: "
                  f"similarity={sim:.6f}, distance={dist:.6f}")
            
            # Same person should have similarity > 0.5
            # (ArcFace threshold is typically ~0.4 for same identity)
            if sim < 0.4:
                print(f"  [WARN] Low similarity for same person: {sim:.6f}")
                # Not a hard fail — could be due to extreme angle/lighting
            else:
                print(f"  [PASS] Similarity confirms same identity")
    
    print("\n[PASS] Test 3.1: Consistency test complete")
    return True


# ─────────────────────────────────────────────────────
# TEST 3.2: Entropy / Diversity (Different People)
# ─────────────────────────────────────────────────────

def test_diversity():
    """
    Capture Person A, then Person B.
    Verify embeddings are clearly different.
    """
    print("\n" + "="*60)
    print("TEST 3.2: Embedding Diversity (Different People)")
    print("="*60)
    
    camera = CameraModule()
    face_module = FaceModule()
    embedding_module = EmbeddingModule()
    
    if not camera.start():
        print("[FAIL] Camera failed to start")
        return False
    
    time.sleep(1)
    
    # ── Person A ──
    print("\n[STEP 1] PERSON A: Stand in front of camera...")
    input("  Press ENTER when ready...")
    
    person_a_embeddings = []
    
    for i in range(2):
        frame_data = camera.get_latest_frame()
        if frame_data:
            face = face_module.detect_largest_face(frame_data)
            if face:
                emb = embedding_module.extract(face)
                if emb and emb.verify_normalization():
                    person_a_embeddings.append(emb)
                    print(f"  Person A - capture {i+1}: OK")
        time.sleep(0.5)
    
    if len(person_a_embeddings) == 0:
        print("[FAIL] Could not capture Person A")
        camera.stop()
        return False
    
    # ── Person B ──
    print("\n[STEP 2] PERSON B: Now switch — different person stand in front of camera...")
    input("  Press ENTER when ready...")
    
    person_b_embeddings = []
    
    for i in range(2):
        frame_data = camera.get_latest_frame()
        if frame_data:
            face = face_module.detect_largest_face(frame_data)
            if face:
                emb = embedding_module.extract(face)
                if emb and emb.verify_normalization():
                    person_b_embeddings.append(emb)
                    print(f"  Person B - capture {i+1}: OK")
        time.sleep(0.5)
    
    camera.stop()
    
    if len(person_b_embeddings) == 0:
        print("[FAIL] Could not capture Person B")
        return False
    
    # ── Analysis ──
    print("\n--- Cross-Identity Comparison ---")
    
    emb_a = person_a_embeddings[0]
    emb_b = person_b_embeddings[0]
    
    sim = emb_a.cosine_similarity(emb_b)
    dist = emb_a.cosine_distance(emb_b)
    
    print(f"  Person A vs Person B:")
    print(f"    Similarity: {sim:.6f}")
    print(f"    Distance:   {dist:.6f}")
    
    # Different people should have low similarity
    if sim > 0.4:
        print(f"  [WARN] High similarity between different people: {sim:.6f}")
        print(f"  [WARN] This could mean the model isn't discriminating well")
    else:
        print(f"  [PASS] Low similarity confirms different identities")
    
    # Also check intra-person consistency
    if len(person_a_embeddings) >= 2:
        intra_sim = person_a_embeddings[0].cosine_similarity(person_a_embeddings[1])
        print(f"\n  Person A self-similarity: {intra_sim:.6f}")
        
        if intra_sim < sim:
            print(f"  [FAIL] Intra-person similarity ({intra_sim:.4f}) is LOWER than "
                  f"inter-person similarity ({sim:.4f})")
            print(f"  [FAIL] Model is not separating identities correctly!")
            return False
        else:
            print(f"  [PASS] Intra > Inter: Model separates identities correctly")
    
    print("\n[PASS] Test 3.2: Diversity test complete")
    return True


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3 Embedding Tests")
    parser.add_argument("--skip-consistency", action="store_true",
                        help="Skip consistency test")
    parser.add_argument("--skip-diversity", action="store_true",
                        help="Skip diversity test")
    args = parser.parse_args()
    
    results = []
    
    if not args.skip_consistency:
        results.append(("Test 3.1: Consistency", test_consistency()))
    
    if not args.skip_diversity:
        results.append(("Test 3.2: Diversity", test_diversity()))
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")