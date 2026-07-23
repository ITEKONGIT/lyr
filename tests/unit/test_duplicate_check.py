"""
Phase 7 - Test 4.1-4.2: Layered Duplicate Check

Tests the three-layer duplicate detection system that prevents
attackers from bypassing the duplicate threshold.

Usage:
    python test_duplicate_check.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import List, Dict


DUPLICATE_THRESHOLD_CENTROID = 0.30
DUPLICATE_THRESHOLD_BEST_FRAME = 0.25
DUPLICATE_THRESHOLD_LOW_QUALITY = 0.35
QUALITY_GRADE_MARGINAL = 0.50


def _unit_vector(dim: int = 512, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _vector_at_distance(base: np.ndarray, target_dist: float) -> np.ndarray:
    """Create a unit vector at a specific cosine distance from base."""
    theta = np.arccos(1.0 - target_dist)
    np.random.seed(42)
    ortho = np.random.randn(512).astype(np.float32)
    ortho -= np.dot(ortho, base) * base
    ortho = ortho / np.linalg.norm(ortho)
    v = base * np.cos(theta) + ortho * np.sin(theta)
    return v / np.linalg.norm(v)


class FakeDB:
    """Minimal database mock for duplicate checking."""

    def __init__(self):
        self.identities = []

    def add(self, user_id: str, name: str, vector: np.ndarray):
        import uuid
        self.identities.append({
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'name': name,
            'vector': vector.copy(),
        })

    def count(self) -> int:
        return len(self.identities)

    def query_nearest(self, vector: np.ndarray) -> List[Dict]:
        if not self.identities:
            return []
        best = None
        best_dist = float('inf')
        for identity in self.identities:
            dist = 1.0 - float(np.dot(vector, identity['vector']))
            if dist < best_dist:
                best_dist = dist
                best = {
                    'id': identity['id'],
                    'user_id': identity['user_id'],
                    'name': identity['name'],
                    'distance': dist,
                    'similarity': 1.0 - dist,
                }
        return [best] if best else []


def find_highest_weighted_frame(
    vectors: list, weights: list
) -> tuple:
    """Find highest-weighted frame index and weight."""
    if not vectors or not weights:
        return 0, 0.0
    max_idx = int(np.argmax(weights))
    return max_idx, weights[max_idx]


def check_duplicate_layered(
    centroid_vector: np.ndarray,
    per_frame_vectors: List[np.ndarray],
    frame_weights: List[float],
    quality_score: float,
    db: FakeDB,
) -> tuple:
    """Three-layer duplicate check."""
    if db.count() == 0:
        return False, None

    centroid_matches = db.query_nearest(centroid_vector)
    best_centroid_match = centroid_matches[0] if centroid_matches else None

    if best_centroid_match and best_centroid_match['distance'] < DUPLICATE_THRESHOLD_CENTROID:
        return True, best_centroid_match

    # Layer 2: Highest-weighted individual frame
    best_frame_idx, _ = find_highest_weighted_frame(
        per_frame_vectors, frame_weights
    )
    best_frame_vector = per_frame_vectors[best_frame_idx]

    frame_matches = db.query_nearest(best_frame_vector)
    if frame_matches and frame_matches[0]['distance'] < DUPLICATE_THRESHOLD_BEST_FRAME:
        return True, frame_matches[0]

    if quality_score < QUALITY_GRADE_MARGINAL:
        if best_centroid_match and best_centroid_match['distance'] < DUPLICATE_THRESHOLD_LOW_QUALITY:
            return True, best_centroid_match

    return False, None


def test_centroid_evasion():
    """
    Database has 'Emmanuel'. The centroid evades Layer 1 by averaging
    noisy frames, but one individual frame is close enough for Layer 2.
    """
    print("\n" + "=" * 60)
    print("TEST 4.1: Centroid Evasion via Layer 2")
    print("=" * 60)

    original = _unit_vector(seed=100)
    db = FakeDB()
    db.add("emmanuel", "Emmanuel", original)

    close_frame = _vector_at_distance(original, 0.20)
    close_dist = 1.0 - float(np.dot(close_frame, original))
    print(f"  Close frame distance to 'Emmanuel': {close_dist:.4f} "
          f"(threshold: {DUPLICATE_THRESHOLD_BEST_FRAME})")

    far_frames = [_vector_at_distance(original, 0.55) for _ in range(4)]

    all_frames = [close_frame] + far_frames
    frame_weights = [2.5] + [1.0] * 4

    centroid = np.mean(all_frames, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    centroid_dist = 1.0 - float(np.dot(centroid, original))
    print(f"  Centroid distance to 'Emmanuel': {centroid_dist:.4f} "
          f"(threshold: {DUPLICATE_THRESHOLD_CENTROID})")

    centroid_matches = db.query_nearest(centroid)
    layer1_hit = (centroid_matches and
                   centroid_matches[0]['distance'] < DUPLICATE_THRESHOLD_CENTROID)
    print(f"  Layer 1 (centroid): hit={layer1_hit}")

    is_dup, match = check_duplicate_layered(
        centroid, all_frames, frame_weights, quality_score=0.85, db=db
    )

    assert not layer1_hit, "Layer 1 should NOT catch at this distance"
    assert is_dup, "Layer 2 should have caught the duplicate"
    assert match is not None
    assert match['name'] == 'Emmanuel'

    print(f"  Caught by: Layer 2 (Best Frame)")
    print(f"  Match: {match['name']} (distance={match['distance']:.4f})")
    print("  [PASS] Centroid evasion correctly caught by best-frame check")
    return True


def test_quality_adjusted_threshold():
    """
    Submit a marginal enrollment (Q=0.45).
    The system should use the relaxed threshold (0.35).
    """
    print("\n" + "=" * 60)
    print("TEST 4.2: Quality-Adjusted Threshold")
    print("=" * 60)

    original = _unit_vector(seed=200)
    db = FakeDB()
    db.add("alice", "Alice", original)

    # All frames at similar distance (0.38), so centroid evades Layer 1
    # and no single frame triggers Layer 2
    frame_dist = 0.38
    all_frames = [_vector_at_distance(original, frame_dist) for _ in range(5)]
    frame_weights = [1.0] * 5

    centroid = np.mean(all_frames, axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    centroid_dist = 1.0 - float(np.dot(centroid, original))
    print(f"  Centroid distance to 'Alice': {centroid_dist:.4f}")
    print(f"  Standard threshold: {DUPLICATE_THRESHOLD_CENTROID}")
    print(f"  Low-quality threshold: {DUPLICATE_THRESHOLD_LOW_QUALITY}")
    print(f"  Quality score: 0.45 (below marginal {QUALITY_GRADE_MARGINAL})")

    is_dup_high_q, _ = check_duplicate_layered(
        centroid, all_frames, frame_weights, quality_score=0.85, db=db
    )
    print(f"  High quality (0.85): duplicate={is_dup_high_q}")

    is_dup_low_q, match = check_duplicate_layered(
        centroid, all_frames, frame_weights, quality_score=0.45, db=db
    )
    print(f"  Low quality (0.45): duplicate={is_dup_low_q}")

    assert not is_dup_high_q, \
        "High quality should pass at distance > 0.30"
    assert is_dup_low_q, \
        "Low quality should trigger tightened check via Layer 3"
    assert match['name'] == 'Alice'

    print("  [PASS] Quality-adjusted threshold correctly catches marginal enrollments")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 7 Duplicate Check Tests"
    )
    parser.add_argument("--skip-evasion", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()

    results = []
    if not args.skip_evasion:
        results.append(("Test 4.1: Centroid Evasion Catch",
                        test_centroid_evasion()))
    if not args.skip_quality:
        results.append(("Test 4.2: Quality-Adjusted Threshold",
                        test_quality_adjusted_threshold()))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: {status}")

    sys.exit(0 if all_pass else 1)
