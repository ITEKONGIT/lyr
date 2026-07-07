"""
Phase 7 ? Test 3.1?3.2: Weighted Centroid Engine

Tests the centroid computation that gives the most mathematically sound
frames the loudest "voice" in the identity baseline.

Usage:
    python test_weighted_centroid.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import List, Tuple
from contracts import EmbeddingData


# ?????????????????????????????????????????????????????
# HELPER
# ?????????????????????????????????????????????????????

def _unit_vector(dim: int = 512, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_emb(seed: int) -> EmbeddingData:
    v = _unit_vector(512, seed)
    return EmbeddingData(vector=v, dimension=512)


# ?????????????????????????????????????????????????????
# CENTROID LOGIC (mirrors RegistrationHandler._compute_weighted_centroid)
# ?????????????????????????????????????????????????????

def compute_weighted_centroid(
    vectors: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1.0] * len(vectors)
        total_weight = len(vectors)

    summed = np.zeros(512, dtype=np.float32)
    for v, w in zip(vectors, weights):
        summed += w * v

    centroid = summed / total_weight
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


# ?????????????????????????????????????????????????????
# TEST 3.1: Garbage-In Resilience
# ?????????????????????????????????????????????????????

def test_garbage_in_resilience():
    """
    Pass 4 excellent vectors and 1 terrible (but passing) vector.
    The weighted centroid should be closer to the 4 excellent frames
    than an unweighted average would be.
    """
    print("\n" + "=" * 60)
    print("TEST 3.1: Garbage-In Resilience")
    print("=" * 60)

    np.random.seed(42)
    base = _unit_vector(512)
    excellent_noise = 0.05

    excellent = []
    for i in range(4):
        v = base + np.random.randn(512).astype(np.float32) * excellent_noise
        excellent.append(v / np.linalg.norm(v))

    terrible = np.random.randn(512).astype(np.float32)
    terrible = terrible / np.linalg.norm(terrible)

    all_vectors = excellent + [terrible]

    # Weighted: excellent get high weights, terrible gets low weight
    excellent_weights = [0.95 * min(120.0 / 40.0, 2.0) for _ in range(4)]
    terrible_weight = [0.50 * min(45.0 / 40.0, 2.0)]  # low c, low v

    all_weights = excellent_weights + terrible_weight
    weighted_centroid = compute_weighted_centroid(all_vectors, all_weights)

    # Unweighted: simple average
    unweighted_centroid = compute_weighted_centroid(
        all_vectors, [1.0] * len(all_vectors)
    )

    # Compare: weighted centroid should be more similar to the excellent set
    excellent_centroid = compute_weighted_centroid(excellent, [1.0] * 4)

    weighted_sim = float(np.dot(weighted_centroid, excellent_centroid))
    unweighted_sim = float(np.dot(unweighted_centroid, excellent_centroid))

    print(f"  Weighted centroid ? excellent: similarity={weighted_sim:.6f}")
    print(f"  Unweighted centroid ? excellent: similarity={unweighted_sim:.6f}")

    assert weighted_sim > unweighted_sim, \
        f"Weighted {weighted_sim:.4f} should exceed unweighted {unweighted_sim:.4f}"

    print("  [PASS] Weighted centroid resists garbage-in better than unweighted")
    return True


# ?????????????????????????????????????????????????????
# TEST 3.2: Unit Sphere Validation
# ?????????????????????????????????????????????????????

def test_unit_sphere_validation():
    """
    Verify that the weighted centroid always lies on the unit sphere
    (L2 norm = 1.0 ? 1e-6).
    """
    print("\n" + "=" * 60)
    print("TEST 3.2: Unit Sphere Validation")
    print("=" * 60)

    np.random.seed(7)

    for trial in range(10):
        n = np.random.randint(2, 8)
        vectors = [_unit_vector(512, seed=i * 100 + trial) for i in range(n)]
        weights = [np.random.uniform(0.5, 2.0) for _ in range(n)]

        centroid = compute_weighted_centroid(vectors, weights)
        norm = float(np.linalg.norm(centroid))

        assert abs(norm - 1.0) < 1e-6, \
            f"Trial {trial}: L2 norm = {norm:.10f}, expected 1.0"

        print(f"  Trial {trial}: {n} vectors, norm={norm:.8f} [PASS]")

    print("\n  [PASS] All centroids remain on unit sphere")
    return True


# ?????????????????????????????????????????????????????
# MAIN
# ?????????????????????????????????????????????????????

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 7 Weighted Centroid Tests"
    )
    parser.add_argument("--skip-resilience", action="store_true")
    parser.add_argument("--skip-unit-sphere", action="store_true")
    args = parser.parse_args()

    results = []
    if not args.skip_resilience:
        results.append(("Test 3.1: Garbage-In Resilience",
                        test_garbage_in_resilience()))
    if not args.skip_unit_sphere:
        results.append(("Test 3.2: Unit Sphere Validation",
                        test_unit_sphere_validation()))

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
