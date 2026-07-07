"""
Phase 4: Database Persistence & Spatial Indexing Module
Stores face embeddings with cosine-similarity nearest-neighbor search.
Uses ChromaDB — local, lightweight, native vector indexing.
"""

import numpy as np
import uuid
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contracts import EmbeddingData, StorageResult


class DatabaseModule:
    """
    Vector database for face embeddings.
    
    Uses ChromaDB with cosine distance metric.
    Each identity stored as: user_id, name, embedding vector, metadata.
    """
    
    def __init__(self, persist_directory: str = None):
        """
        Args:
            persist_directory: Where to store the database files.
                              Defaults to ./face_db relative to this file.
        """
        if persist_directory is None:
            persist_directory = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "face_db"
            )
        
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None
        
        self._initialize()
    
    def _initialize(self):
        """Set up ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection with cosine distance
            self.collection = self.client.get_or_create_collection(
                name="face_identities",
                metadata={"hnsw:space": "cosine"}
            )
            
            count = self.collection.count()
            print(f"[DatabaseModule] Initialized at: {self.persist_directory}")
            print(f"[DatabaseModule] Collection: face_identities")
            print(f"[DatabaseModule] Existing records: {count}")
            print(f"[DatabaseModule] Distance metric: cosine")
            
        except ImportError:
            print("[DatabaseModule] ERROR: chromadb not installed.")
            print("[DatabaseModule] Install with: pip install chromadb")
            raise
        except Exception as e:
            print(f"[DatabaseModule] ERROR initializing: {e}")
            raise
    
    # ──────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────
    
    def store(self, user_id: str, name: str, 
              embedding: EmbeddingData,
              metadata: Optional[Dict] = None) -> StorageResult:
        """
        Store a face embedding with identity information.
        
        Args:
            user_id: Unique user identifier (your system's ID)
            name: Human-readable name
            embedding: EmbeddingData from Phase 3
            metadata: Optional additional metadata dict
            
        Returns:
            StorageResult with document ID and roundtrip verification
        """
        start_time = time.time()
        
        doc_id = str(uuid.uuid4())
        
        full_metadata = {
            "user_id": user_id,
            "name": name,
            "model": embedding.model_name,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "ema_count": 0
        }
        if metadata:
            full_metadata.update(metadata)
        
        # Store in ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.vector.tolist()],
            documents=[name],
            metadatas=[full_metadata]
        )
        
        # Roundtrip: retrieve immediately to verify
        retrieved = self.collection.get(
            ids=[doc_id],
            include=["embeddings", "metadatas", "documents"]
        )
        
        retrieved_vector = None
        if retrieved['embeddings'] is not None and len(retrieved['embeddings']) > 0:
            retrieved_vector = np.array(retrieved['embeddings'][0], dtype=np.float32)
        
        query_time = (time.time() - start_time) * 1000
        
        result = StorageResult(
            document_id=doc_id,
            user_id=user_id,
            name=name,
            stored_vector=embedding.vector,
            retrieved_vector=retrieved_vector,
            query_latency_ms=round(query_time, 2),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        print(f"[DatabaseModule] Stored: {name} (id={doc_id[:8]}...) "
              f"in {query_time:.1f}ms")
        
        return result
    
    def update_vector(self, doc_id: str, new_vector: np.ndarray, 
                      ema_alpha: float = 0.1) -> Tuple[bool, Optional[Dict]]:
        """
        Update a stored embedding using Exponential Moving Average (EMA).
        
        Blends the existing vector with the new vector:
            updated = (1 - alpha) * existing + alpha * new
        
        Then re-normalizes to unit length. Used for adaptive identity
        learning — the system gradually adapts to lighting changes,
        haircuts, aging, etc.
        
        Args:
            doc_id: Document ID to update
            new_vector: New live embedding (must be L2-normalized)
            ema_alpha: Blend weight for new vector [0.0, 1.0]
                       Higher = faster adaptation
                       
        Returns:
            Tuple of (success: bool, result_dict: Optional[Dict])
            result_dict contains: old_norm, new_norm, drift_distance, ema_count
        """
        # Retrieve existing
        existing = self.get_by_id(doc_id)
        if existing is None:
            print(f"[DatabaseModule] update_vector: ID {doc_id[:8]}... not found")
            return False, None
        
        old_vector = existing['embedding']
        old_metadata = existing['metadata']
        
        # Validate new_vector
        new_norm = np.linalg.norm(new_vector)
        if abs(new_norm - 1.0) > 0.01:
            # Re-normalize if needed
            new_vector = new_vector / new_norm
            print(f"[DatabaseModule] update_vector: re-normalized new vector "
                  f"(was {new_norm:.4f})")
        
        # EMA blend
        blended = (1.0 - ema_alpha) * old_vector + ema_alpha * new_vector
        
        # Re-normalize to unit length
        blended_norm = np.linalg.norm(blended)
        blended = blended / blended_norm
        
        # Track how many times this identity has been updated
        ema_count = old_metadata.get('ema_count', 0) + 1
        
        # Measure drift from original
        drift = float(np.linalg.norm(old_vector - blended))
        
        # Update metadata
        updated_metadata = {**old_metadata}
        updated_metadata['ema_count'] = ema_count
        updated_metadata['last_ema_update'] = datetime.now(timezone.utc).isoformat()
        updated_metadata['last_drift'] = round(drift, 6)
        
        # ChromaDB doesn't support direct update — we delete and re-add
        try:
            self.collection.delete(ids=[doc_id])
            self.collection.add(
                ids=[doc_id],
                embeddings=[blended.tolist()],
                documents=[existing['name']],
                metadatas=[updated_metadata]
            )
            
            result = {
                'doc_id': doc_id,
                'name': existing['name'],
                'old_norm': round(float(np.linalg.norm(old_vector)), 4),
                'new_norm': round(float(blended_norm), 4),
                'drift_distance': round(drift, 6),
                'ema_alpha': ema_alpha,
                'ema_count': ema_count
            }
            
            print(f"[DatabaseModule] EMA update: {existing['name']} "
                  f"(count={ema_count}, drift={drift:.6f}, alpha={ema_alpha})")
            
            return True, result
            
        except Exception as e:
            print(f"[DatabaseModule] update_vector failed: {e}")
            return False, None
    
    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a stored identity by document ID.
        
        Returns:
            Dict with id, user_id, name, embedding, metadata, or None
        """
        result = self.collection.get(
            ids=[doc_id],
            include=["embeddings", "metadatas", "documents"]
        )
        
        if not result['ids']:
            return None
        
        return {
            'id': result['ids'][0],
            'user_id': result['metadatas'][0].get('user_id', ''),
            'name': result['documents'][0],
            'embedding': np.array(result['embeddings'][0], dtype=np.float32),
            'metadata': result['metadatas'][0]
        }
    
    def get_by_user_id(self, user_id: str) -> List[Dict]:
        """
        Retrieve all stored identities for a given user_id.
        """
        results = self.collection.get(
            where={"user_id": user_id},
            include=["embeddings", "metadatas", "documents"]
        )
        
        identities = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids']):
                identities.append({
                    'id': doc_id,
                    'user_id': results['metadatas'][i].get('user_id', ''),
                    'name': results['documents'][i],
                    'embedding': np.array(results['embeddings'][i], dtype=np.float32),
                    'metadata': results['metadatas'][i]
                })
        
        return identities
    
    def query_nearest(self, embedding: EmbeddingData, 
                      n_results: int = 5) -> List[Dict]:
        """
        Find the nearest neighbors to a query embedding.
        
        Uses cosine distance — lower distance = more similar.
        
        Args:
            embedding: Query embedding
            n_results: Number of nearest neighbors to return
            
        Returns:
            List of dicts with id, user_id, name, distance, similarity, metadata
        """
        start_time = time.time()
        
        results = self.collection.query(
            query_embeddings=[embedding.vector.tolist()],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        query_time = (time.time() - start_time) * 1000
        
        matches = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                matches.append({
                    'id': doc_id,
                    'user_id': results['metadatas'][0][i].get('user_id', ''),
                    'name': results['documents'][0][i],
                    'distance': distance,
                    'similarity': 1.0 - distance,
                    'metadata': results['metadatas'][0][i]
                })
        
        print(f"[DatabaseModule] Query returned {len(matches)} results "
              f"in {query_time:.1f}ms")
        
        return matches
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a stored identity by document ID.
        
        Returns:
            True if deleted successfully
        """
        try:
            self.collection.delete(ids=[doc_id])
            print(f"[DatabaseModule] Deleted: {doc_id}")
            return True
        except Exception as e:
            print(f"[DatabaseModule] Delete failed: {e}")
            return False
    
    def count(self) -> int:
        """Return total number of stored identities."""
        return self.collection.count()
    
    def list_all(self) -> List[Dict]:
        """List all stored identities (without embeddings for efficiency)."""
        results = self.collection.get(
            include=["metadatas", "documents"]
        )
        
        identities = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids']):
                identities.append({
                    'id': doc_id,
                    'user_id': results['metadatas'][i].get('user_id', ''),
                    'name': results['documents'][i],
                    'stored_at': results['metadatas'][i].get('stored_at', '')
                })
        
        return identities
    
    def reset(self) -> None:
        """
        Delete ALL stored identities. Use with caution.
        """
        count = self.collection.count()
        if count > 0:
            all_ids = self.collection.get()['ids']
            self.collection.delete(ids=all_ids)
            print(f"[DatabaseModule] Reset: deleted {count} records")
        else:
            print("[DatabaseModule] Reset: database already empty")