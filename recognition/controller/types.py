"""
Internal types shared across controller components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from contracts import EmbeddingData


@dataclass
class PendingRequest:
    """A queued registration or identification request."""
    action: str             # "register"
    user_id: str
    name: str
    metadata: Dict
    queued_at: float


@dataclass
class DBTask:
    """A task for the database worker thread."""
    action: str             # "register"
    request_id: str
    user_id: str
    name: str
    embedding: EmbeddingData
    metadata: Dict
    liveness: Dict
    processing_time_ms: float