"""
Phase 7 - Test 1.1-1.3: Frame Weight Formula & Lighting Gate

Tests the core mathematical weighting logic that assesses individual
frames before they influence the centroid.

Usage:
    python test_quality_evaluator.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from contracts import LightingVarianceError


THRESHOLD = 40.0


def compute_frame_weight(confidence: float, variance: float,
                         threshold: float = THRESHOLD) -> float:
    """Frame weight formula: w = c * min(v / tau, 2.0)"""
    if variance < threshold:
        return 0.0
    liveness_factor = min(variance / threshold, 2.0)
    return confidence * liveness_factor


def test_optimal_frame():
    """
    Pass a frame with c=0.98 and v=120 (tau=40.0).
    Expected weight = 1.96 (hits the 2.0 cap multiplier).
    """
    print("\n" + "=" * 60)
    print("TEST 1.1: Optimal Frame Weight")
    print("=" * 60)

    confidence = 0.98
    variance = 120.0

    weight = compute_frame_weight(confidence, variance)
    expected = 1.96

    print(f"  confidence={confidence}, variance={variance}, tau={THRESHOLD}")
    print(f"  weight={weight:.4f} (expected={expected:.4f})")

    assert abs(weight - expected) < 1e-4, \
        f"Weight {weight:.4f} != expected {expected:.4f}"
    assert weight <= 2.0, f"Weight {weight:.4f} exceeded cap of 2.0"

    print("  [PASS] Weight correctly hits cap multiplier")
    return True


def test_marginal_frame():
    """
    Pass a frame with c=0.60 and v=45 (just above tau).
    Expected weight = 0.60 * 45/40 = 0.675 (no cap hit).
    """
    print("\n" + "=" * 60)
    print("TEST 1.2: Marginal Frame Weight")
    print("=" * 60)

    confidence = 0.60
    variance = 45.0

    weight = compute_frame_weight(confidence, variance)
    expected = 0.675

    print(f"  confidence={confidence}, variance={variance}, tau={THRESHOLD}")
    print(f"  weight={weight:.4f} (expected={expected:.4f})")

    assert abs(weight - expected) < 1e-4, \
        f"Weight {weight:.4f} != expected {expected:.4f}"
    assert weight < 2.0, f"Weight {weight:.4f} should not hit cap"

    print("  [PASS] Weight proportional and below cap")
    return True


def test_lighting_gate_rejection():
    """
    Pass a frame with v=25 (below tau=40.0).
    Expected: weight = 0.0 (lighting gate reject).
    """
    print("\n" + "=" * 60)
    print("TEST 1.3: Lighting Gate Rejection")
    print("=" * 60)

    confidence = 0.90
    variance = 25.0

    weight = compute_frame_weight(confidence, variance)
    expected = 0.0

    print(f"  confidence={confidence}, variance={variance}, tau={THRESHOLD}")
    print(f"  weight={weight:.4f} (expected={expected:.4f})")

    assert weight == 0.0, f"Weight {weight} should be 0.0 for low variance"

    error = LightingVarianceError(variance=variance, threshold=THRESHOLD)
    print(f"  LightingVarianceError: {error}")

    assert error.variance == variance
    assert error.threshold == THRESHOLD

    print("  [PASS] Lighting gate correctly assigns weight=0.0")
    return True


if __name__ == "__main__":
    results = [
        ("Test 1.1: Optimal Frame", test_optimal_frame()),
        ("Test 1.2: Marginal Frame", test_marginal_frame()),
        ("Test 1.3: Lighting Gate Rejection", test_lighting_gate_rejection()),
    ]

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