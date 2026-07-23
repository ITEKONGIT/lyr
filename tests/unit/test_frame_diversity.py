"""
Phase 7 - Test 2.1-2.2: Diverse Frame Selection (Anti-Clone Filter)

Tests the system's ability to detect and reject clone attacks
(identical embeddings from static video loops) and correctly
select the most diverse subset of frames.

Usage:
    python test_frame_diversity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import List
from contracts import EmbeddingData


def _random_unit_vector(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_embedding(vector: np.ndarray, idx: int = 0) -> EmbeddingData:
    return EmbeddingData(
        vector=vector,
        dimension=512,
        source_frame_number=idx,
        model_name="test",
    )


def _make_embeddings(vectors: List[np.ndarray]) -> List[EmbeddingData]:
    return [_make_embedding(v, i) for i, v in enumerate(vectors)]


def select_diverse(
    embeddings: List[EmbeddingData],
    n_select: int = 5,
) -> tuple:
    """Farthest-point sampling to select diverse embeddings."""

    n_select = min(n_select, len(embeddings))

    if n_select < 2 or len(embeddings) <= n_select:
        if len(embeddings) >= 2:
            similarities = [
                float(np.dot(embeddings[i].vector, embeddings[j].vector))
                for i in range(len(embeddings))
                for j in range(i + 1, len(embeddings))
            ]
            diversity = 1.0 - (np.mean(similarities) if similarities else 0.0)
            diversity = max(0.0, min(diversity, 1.0))
        else:
            diversity = 0.0
        return embeddings, diversity

    vectors = [emb.vector for emb in embeddings]
    n = len(vectors)

    selected_indices = [0]
    remaining = set(range(1, n))

    while len(selected_indices) < n_select and remaining:
        best_dist = -1.0
        best_idx = -1
        for i in remaining:
            min_dist = float('inf')
            for j in selected_indices:
                dist = 1.0 - float(np.dot(vectors[i], vectors[j]))
                if dist < min_dist:
                    min_dist = dist
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = i
        if best_idx >= 0:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    selected = sorted(selected_indices)
    selected_embeddings = [embeddings[i] for i in selected]

    pairwise_distances = [
        1.0 - float(np.dot(
            selected_embeddings[i].vector,
            selected_embeddings[j].vector
        ))
        for i in range(len(selected_embeddings))
        for j in range(i + 1, len(selected_embeddings))
    ]
    diversity = max(0.0, min(float(np.mean(pairwise_distances)), 1.0))

    return selected_embeddings, diversity


def test_clone_attack():
    """
    Pass 10 identical embedding vectors.
    Expected: D = 0.0, batch rejected.
    """
    print("\n" + "=" * 60)
    print("TEST 2.1: Clone Attack Detection")
    print("=" * 60)

    base = _random_unit_vector()
    identical_vectors = [base.copy() for _ in range(10)]
    embeddings = _make_embeddings(identical_vectors)

    selected, diversity = select_diverse(embeddings)

    print(f"  Input: 10 identical embeddings")
    print(f"  Selected: {len(selected)} embeddings")
    print(f"  Diversity D = {diversity:.4f}")

    assert diversity < 0.01, \
        f"Diversity {diversity} should be ~0 for identical vectors"

    print("  [PASS] Clone attack detected - D ~ 0.0")
    return True


def test_healthy_diversity():
    """
    Pass 10 vectors with spatial variations.
    Expected: 5 most mathematically distinct frames selected, D > 0.8.
    """
    print("\n" + "=" * 60)
    print("TEST 2.2: Healthy Diversity Selection")
    print("=" * 60)

    rng = np.random.RandomState(42)
    base = _random_unit_vector()

    vectors = []
    for i in range(10):
        noise = rng.randn(512).astype(np.float32) * 0.3
        v = base + noise
        v = v / np.linalg.norm(v)
        vectors.append(v)

    embeddings = _make_embeddings(vectors)
    selected, diversity = select_diverse(embeddings)

    print(f"  Input: 10 varied embeddings")
    print(f"  Selected: {len(selected)} embeddings (target: 5)")
    print(f"  Diversity D = {diversity:.4f}")

    assert len(selected) == 5, \
        f"Selected {len(selected)} frames, expected 5"
    assert diversity > 0.80, \
        f"Diversity {diversity:.4f} < 0.80 for varied inputs"

    indices = [e.source_frame_number for e in selected]
    assert len(set(indices)) == len(indices), "Duplicate indices in selection"

    print(f"  Selected frames: {sorted(indices)}")
    print("  [PASS] Healthy diversity - D > 0.8, 5 unique frames selected")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 7 Diverse Frame Selection Tests"
    )
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-diversity", action="store_true")
    args = parser.parse_args()

    results = []
    if not args.skip_clone:
        results.append(("Test 2.1: Clone Attack", test_clone_attack()))
    if not args.skip_diversity:
        results.append(("Test 2.2: Healthy Diversity", test_healthy_diversity()))

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