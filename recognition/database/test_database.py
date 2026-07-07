"""
Phase 4 Verification Tests

Test 4.1 (Storage Verification): Write + retrieve, verify bit-for-bit match
Test 4.2 (Distance Query): Populate 10 identities, nearest-neighbor accuracy + speed

Usage:
    python database/test_database.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from contracts import EmbeddingData, StorageResult
from database.database_module import DatabaseModule


# ─────────────────────────────────────────────────────
# TEST 4.1: Storage Verification (Bit-for-Bit)
# ─────────────────────────────────────────────────────

def test_storage_roundtrip():
    """
    Write a mock identity, retrieve it, verify bit-for-bit match.
    """
    print("\n" + "="*60)
    print("TEST 4.1: Storage Verification (Roundtrip)")
    print("="*60)
    
    db = DatabaseModule(persist_directory="./test_db_4_1")
    db.reset()  # Start clean
    
    # Create a known embedding
    vector = np.random.randn(512).astype(np.float32)
    vector = vector / np.linalg.norm(vector)  # L2 normalize
    
    embedding = EmbeddingData(
        vector=vector,
        dimension=512,
        source_frame_number=42,
        model_name="test_model"
    )
    
    # Store
    print("\n--- Storing ---")
    result = db.store(
        user_id="test_user_001",
        name="Test Person",
        embedding=embedding,
        metadata={"test_key": "test_value"}
    )
    
    # Verify roundtrip
    if not result.verify_roundtrip():
        print("[FAIL] Roundtrip verification failed")
        return False
    
    print(f"  Document ID: {result.document_id}")
    print(f"  Query latency: {result.query_latency_ms:.1f}ms")
    print(f"  Vectors match: bit-for-bit identical")
    
    # Retrieve by ID
    print("\n--- Retrieving by ID ---")
    identity = db.get_by_id(result.document_id)
    
    if identity is None:
        print("[FAIL] Could not retrieve by ID")
        return False
    
    # Verify retrieved data
    if identity['user_id'] != "test_user_001":
        print(f"[FAIL] Wrong user_id: {identity['user_id']}")
        return False
    
    if identity['name'] != "Test Person":
        print(f"[FAIL] Wrong name: {identity['name']}")
        return False
    
    if not np.array_equal(identity['embedding'], vector):
        print("[FAIL] Retrieved vector doesn't match original")
        return False
    
    print(f"  Retrieved: {identity['name']} (user_id={identity['user_id']})")
    print(f"  Metadata: {identity['metadata']}")
    print(f"  Vector shape: {identity['embedding'].shape}")
    print(f"  Vector norm: {np.linalg.norm(identity['embedding']):.4f}")
    
    # Test retrieval by user_id
    print("\n--- Retrieving by user_id ---")
    identities = db.get_by_user_id("test_user_001")
    
    if len(identities) != 1:
        print(f"[FAIL] Expected 1 identity, got {len(identities)}")
        return False
    
    print(f"  Found {len(identities)} identity for user_id=test_user_001")
    
    # Test deletion
    print("\n--- Deleting ---")
    db.delete(result.document_id)
    
    identity = db.get_by_id(result.document_id)
    if identity is not None:
        print("[FAIL] Identity still exists after deletion")
        return False
    
    print("  Identity deleted and verified gone")
    
    # Cleanup
    db.reset()
    
    print("\n[PASS] Test 4.1: Storage verification passed")
    return True


# ─────────────────────────────────────────────────────
# TEST 4.2: Nearest-Neighbor Query (Accuracy + Speed)
# ─────────────────────────────────────────────────────

def test_nearest_neighbor():
    """
    Populate database with 10 mock identities.
    Query with the exact vector of identity #1.
    Verify:
    - Identity #1 is returned as closest match
    - Query completes in milliseconds
    - Distance to self is near-zero
    """
    print("\n" + "="*60)
    print("TEST 4.2: Nearest-Neighbor Query")
    print("="*60)
    
    db = DatabaseModule(persist_directory="./test_db_4_2")
    db.reset()
    
    # Create 10 mock identities with random vectors
    mock_vectors = []
    mock_ids = []
    
    for i in range(10):
        vector = np.random.randn(512).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        mock_vectors.append(vector)
        
        embedding = EmbeddingData(
            vector=vector,
            dimension=512,
            source_frame_number=i,
            model_name="test_model"
        )
        
        user_id = f"user_{i:03d}"
        name = f"Person {i}"
        
        result = db.store(
            user_id=user_id,
            name=name,
            embedding=embedding
        )
        mock_ids.append(result.document_id)
        
        print(f"  Stored: {name} (id={result.document_id[:8]}...)")
    
    print(f"\n  Database now has {db.count()} identities")
    
    # Query with the exact vector of Person 0
    print("\n--- Querying with Person 0's vector ---")
    
    query_embedding = EmbeddingData(
        vector=mock_vectors[0],
        dimension=512,
        source_frame_number=999,
        model_name="test_model"
    )
    
    start_time = time.time()
    results = db.query_nearest(query_embedding, n_results=3)
    query_time = (time.time() - start_time) * 1000
    
    # Verify results
    if len(results) == 0:
        print("[FAIL] No results returned")
        return False
    
    print(f"\n  Top 3 matches:")
    for i, match in enumerate(results):
        print(f"    {i+1}. {match['name']} — "
              f"distance={match['distance']:.6f}, "
              f"similarity={match['similarity']:.6f}")
    
    # Check: Person 0 should be the closest match
    closest = results[0]
    
    if closest['name'] != "Person 0":
        print(f"[FAIL] Expected 'Person 0' as closest, got '{closest['name']}'")
        return False
    
    if closest['distance'] > 0.001:
        print(f"[FAIL] Distance to self should be near-zero, "
              f"got {closest['distance']:.6f}")
        return False
    
    print(f"\n  Closest match is correct: {closest['name']}")
    print(f"  Self-distance: {closest['distance']:.8f} (expected ~0.0)")
    print(f"  Query latency: {query_time:.1f}ms")
    
    # Check: query completes in reasonable time
    if query_time > 500:
        print(f"[WARN] Query took {query_time:.1f}ms — should be under 500ms")
    
    # Also test: Person 1's vector should NOT be closest to Person 0
    query_emb_1 = EmbeddingData(
        vector=mock_vectors[1],
        dimension=512,
        model_name="test_model"
    )
    results_1 = db.query_nearest(query_emb_1, n_results=1)
    
    if results_1[0]['name'] != "Person 1":
        print(f"[FAIL] Wrong match for Person 1: {results_1[0]['name']}")
        return False
    
    print(f"  Person 1 correctly matches self: {results_1[0]['name']}")
    
    # Cleanup
    db.reset()
    
    print("\n[PASS] Test 4.2: Nearest-neighbor query passed")
    return True


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 4 Database Tests")
    parser.add_argument("--skip-storage", action="store_true",
                        help="Skip storage verification test")
    parser.add_argument("--skip-query", action="store_true",
                        help="Skip nearest-neighbor query test")
    args = parser.parse_args()
    
    results = []
    
    if not args.skip_storage:
        results.append(("Test 4.1: Storage Verification", 
                        test_storage_roundtrip()))
    
    if not args.skip_query:
        results.append(("Test 4.2: Nearest-Neighbor Query", 
                        test_nearest_neighbor()))
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")