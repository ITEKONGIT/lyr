"""
Database Worker Thread

Sequential database writer. Single-threaded by design to prevent
SQLite concurrency locks. Processes one write at a time.
"""

import queue
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Dict

from database.database_module import DatabaseModule
from contracts import RegistrationResult


class DatabaseWorker:
    """
    Single-responsibility: process DB tasks sequentially.
    """
    
    def __init__(self, database: DatabaseModule, db_queue: queue.Queue):
        self.database = database
        self.db_queue = db_queue
        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        
        # Shared state with orchestrator
        self.registrations_completed = 0
        self.results: Dict[str, RegistrationResult] = {}
        self._lock = threading.Lock()
    
    def start(self):
        """Start the worker thread."""
        if self._is_running:
            return
        
        self._is_running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="DatabaseWorker"
        )
        self._thread.start()
        print("[DatabaseWorker] Started")
    
    def stop(self):
        """Signal the thread to stop."""
        self._is_running = False
        try:
            self.db_queue.put_nowait(None)
        except queue.Full:
            pass
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                print("[DatabaseWorker] WARNING: did not exit cleanly")
        print("[DatabaseWorker] Stopped")
    
    def store_result(self, request_id: str, result: RegistrationResult):
        """Store a result for the orchestrator to retrieve."""
        with self._lock:
            self.results[request_id] = result
    
    def get_result(self, request_id: str) -> Optional[RegistrationResult]:
        """Retrieve a stored result."""
        with self._lock:
            return self.results.pop(request_id, None)
    
    def _loop(self):
        """Main worker loop."""
        while self._is_running:
            try:
                task = self.db_queue.get(timeout=0.5)
                
                if task is None:
                    break
                
                if task['action'] == 'register':
                    self._process_registration(task)
                else:
                    print(f"[DatabaseWorker] Unknown action: {task['action']}")
                
                self.db_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DatabaseWorker] Error: {e}")
                continue
    
    def _process_registration(self, task: Dict):
        """Store a face embedding in the database."""
        request_id = task['request_id']
        user_id = task['user_id']
        name = task['name']
        embedding = task['embedding']
        metadata = task.get('metadata', {})
        liveness_info = task.get('liveness', {})
        processing_time = task.get('processing_time_ms', 0.0)
        
        storage_result = self.database.store(
            user_id=user_id,
            name=name,
            embedding=embedding,
            metadata=metadata
        )
        
        if storage_result.verify_roundtrip():
            self.registrations_completed += 1
            result = RegistrationResult(
                status="success",
                transaction_id=storage_result.document_id,
                user_id=user_id,
                name=name,
                liveness=liveness_info,
                embedding_dim=512,
                processing_time_ms=processing_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        else:
            result = RegistrationResult(
                status="failed",
                user_id=user_id,
                name=name,
                liveness=liveness_info,
                error="Database roundtrip verification failed",
                processing_time_ms=processing_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        self.store_result(request_id, result)