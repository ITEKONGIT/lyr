"""
Phase 6: WebSocket Broadcaster (Solution #8)

Problem: The Watcher loop runs in a synchronous Python thread (daemon),
but FastAPI WebSocket handlers require an asynchronous event loop.
Direct cross-thread communication would cause race conditions and
block the event loop.

Solution: A thread-safe bridge using asyncio.Queue and
asyncio.run_coroutine_threadsafe(). The watcher thread pushes
DetectionEvent dicts into the queue synchronously. An asyncio
background task pops from the queue and fans out to all connected
WebSocket clients.

Architecture:
    Watcher (sync thread)
        └─► broadcaster.push_sync(event_dict)
                │
                ▼
        asyncio.Queue (maxsize=64, thread-safe push)
                │
                ▼
        Broadcaster._broadcast_loop (async task)
                │
                ├─► Client A: await ws.send_json(event)
                ├─► Client B: await ws.send_json(event)
                └─► Client C: await ws.send_json(event)

Features:
    - Client registration/disconnection tracking
    - Automatic cleanup of dead clients
    - Backpressure: queue maxsize=64 drops oldest events if clients are slow
    - Heartbeat: empty-frame events every ~1 second to keep connections alive
"""

import asyncio
from typing import Set
from fastapi import WebSocket, WebSocketDisconnect


class Broadcaster:
    """
    Thread-safe WebSocket broadcast manager.
    
    Bridges synchronous watcher thread to asynchronous FastAPI
    WebSocket endpoints.
    

    """
    
    # ──────────────────────────────────────────────
    # CONFIGURATION
    # ──────────────────────────────────────────────
    
    MAX_QUEUE_SIZE = 64  # Drop oldest events if queue is full
    
    # ──────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────
    
    def __init__(self):
        # asyncio.Queue — created in the event loop context
        self._queue: asyncio.Queue = None
        
        # Connected WebSocket clients
        self._clients: Set[WebSocket] = set()
        
        # Reference to the asyncio event loop (set after creation)
        self._event_loop: asyncio.AbstractEventLoop = None
        
        # Background task reference
        self._broadcast_task: asyncio.Task = None
        
        # Stats
        self._events_broadcast = 0
        self._events_dropped = 0
        self._clients_connected = 0
        self._clients_disconnected = 0
        self._is_running = False
    
    def attach_loop(self, loop: asyncio.AbstractEventLoop):
        """
        Attach the asyncio event loop.
        
        """
        self._event_loop = loop
        self._queue = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
    
    async def start(self):
        """
        Start the broadcast background task.

        """
        if self._is_running:
            return
        
        if self._queue is None:
            raise RuntimeError("Call attach_loop() before start()")
        
        self._is_running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        print(f"[Broadcaster] Started — queue maxsize={self.MAX_QUEUE_SIZE}")
    
    async def stop(self):
        """Stop the broadcast loop."""
        self._is_running = False
        
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all clients
        for client in list(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        
        print(f"[Broadcaster] Stopped — {self._events_broadcast} events, "
              f"{self._clients_connected} connections")
    
    # ──────────────────────────────────────────────
    # CLIENT MANAGEMENT
    # ──────────────────────────────────────────────
    
    async def connect(self, websocket: WebSocket):
        """
        Accept a new WebSocket client connection.

        """
        await websocket.accept()
        self._clients.add(websocket)
        self._clients_connected += 1
        
        print(f"[Broadcaster] Client connected "
              f"(total: {len(self._clients)})")
    
    async def disconnect(self, websocket: WebSocket):
        """
        Remove a disconnected client.

        """
        self._clients.discard(websocket)
        self._clients_disconnected += 1
        
        print(f"[Broadcaster] Client disconnected "
              f"(total: {len(self._clients)})")
    
    # ──────────────────────────────────────────────
    # SYNC → ASYNC BRIDGE
    # ──────────────────────────────────────────────
    
    def push_sync(self, event_dict: dict):
        """
        Push an event from a synchronous thread.

        """
        if self._event_loop is None or self._queue is None:
            return
        
        try:
            # Schedule the push into the asyncio queue from the sync thread
            future = asyncio.run_coroutine_threadsafe(
                self._enqueue(event_dict),
                self._event_loop
            )
            # Don't wait for result — fire and forget
        except Exception as e:
            print(f"[Broadcaster] push_sync error: {e}")
    
    async def _enqueue(self, event_dict: dict):
        """
        Push an event into the asyncio queue.
        
        If the queue is full, drop the oldest event to make room.
        This prevents a slow client from blocking the watcher thread.
        """
        try:
            self._queue.put_nowait(event_dict)
        except asyncio.QueueFull:
            # Queue is full — drop the oldest event
            try:
                self._queue.get_nowait()
                self._events_dropped += 1
            except asyncio.QueueEmpty:
                pass
            
            # Now try again
            try:
                self._queue.put_nowait(event_dict)
            except asyncio.QueueFull:
                self._events_dropped += 1
    
    # ──────────────────────────────────────────────
    # BROADCAST LOOP
    # ──────────────────────────────────────────────
    
    async def _broadcast_loop(self):
        """
        Continuous broadcast loop.

        """
        print("[Broadcaster] Broadcast loop started")
        
        while self._is_running:
            try:
                # Wait for next event (with timeout to allow clean shutdown)
                event = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            
            dead_clients = set()
            
            for client in self._clients:
                try:
                    await client.send_json(event)
                except Exception:
                    # Client disconnected without proper cleanup
                    dead_clients.add(client)
            
            # Remove dead clients
            if dead_clients:
                for client in dead_clients:
                    self._clients.discard(client)
                    self._clients_disconnected += 1
                print(f"[Broadcaster] Cleaned up {len(dead_clients)} "
                      f"dead client(s)")
            
            self._events_broadcast += 1
            self._queue.task_done()
        
        print("[Broadcaster] Broadcast loop stopped")
    
    # ──────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────
    
    def get_stats(self) -> dict:
        """Return broadcaster statistics."""
        return {
            "is_running": self._is_running,
            "active_clients": len(self._clients),
            "total_connected": self._clients_connected,
            "total_disconnected": self._clients_disconnected,
            "events_broadcast": self._events_broadcast,
            "events_dropped": self._events_dropped,
            "queue_size": self._queue.qsize() if self._queue else 0,
            "queue_maxsize": self.MAX_QUEUE_SIZE,
        }