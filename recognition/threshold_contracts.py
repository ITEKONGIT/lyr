"""
Tier 2 threshold-gate data contracts.

This module defines declarative rule and breach data structures only. It does
not evaluate readings or update breach state.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from .sensor_contracts import SensorType


DEFAULT_CROSS_SENSOR_HISTORY_SECONDS = 60
DEFAULT_CLUSTER_WINDOW_MS = 500
DEFAULT_BASE_CONFIDENCE = 0.50
DEFAULT_MAX_CONFIDENCE = 0.95
DEFAULT_MISSING_REQUIRED_PENALTY = 0.15
DEFAULT_STALE_REQUIRED_PENALTY = 0.15


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class RuleSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class RuleMode(str, Enum):
    LOG_ONLY = "log_only"
    ESCALATE = "escalate"


class StalenessPolicy(str, Enum):
    ALERT_STALE = "alert_stale"
    FAIL_CLOSED = "fail_closed"
    FAIL_OPEN = "fail_open"


class BreachStatus(str, Enum):
    IDLE = "idle"
    PENDING_SUSTAIN = "pending_sustain"
    ACTIVE = "active"
    CLEARING = "clearing"
    CLEARED = "cleared"


class ComparisonOperator(str, Enum):
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    RISING = "rising"
    DROPPING = "dropping"


class ContextEffect(str, Enum):
    NEUTRAL = "neutral"
    SUPPRESS = "suppress"
    ESCALATE = "escalate"


class EvidenceRole(str, Enum):
    CORROBORATES = "corroborates"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"


@dataclass
class RuleCondition:
    """
    One declarative predicate over a sensor value or derived history trend.
    """

    sensor_type: SensorType
    operator: ComparisonOperator
    threshold: float
    sensor_id: Optional[str] = None
    history_window_seconds: int = DEFAULT_CROSS_SENSOR_HISTORY_SECONDS
    required: bool = True
    effect: ContextEffect = ContextEffect.NEUTRAL
    role: EvidenceRole = EvidenceRole.CORROBORATES
    weight: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.sensor_type, str):
            self.sensor_type = SensorType.from_string(self.sensor_type)
        if isinstance(self.operator, str):
            try:
                self.operator = ComparisonOperator(self.operator)
            except ValueError as exc:
                raise ValueError(f"Unsupported comparison operator: {self.operator}") from exc
        if self.history_window_seconds < 0:
            raise ValueError("history_window_seconds cannot be negative")
        if isinstance(self.effect, str):
            try:
                self.effect = ContextEffect(self.effect)
            except ValueError as exc:
                raise ValueError(f"Unsupported context effect: {self.effect}") from exc
        if isinstance(self.role, str):
            try:
                self.role = EvidenceRole(self.role)
            except ValueError as exc:
                raise ValueError(f"Unsupported evidence role: {self.role}") from exc
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError("condition weight must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor_type": self.sensor_type.value,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "sensor_id": self.sensor_id,
            "history_window_seconds": self.history_window_seconds,
            "required": self.required,
            "effect": self.effect.value,
            "role": self.role.value,
            "weight": self.weight,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleCondition":
        return cls(**data)


@dataclass
class Rule:
    """
    Declarative threshold rule.

    Phase 2.1 defines validation and serialization only. Evaluation arrives in
    later phases.
    """

    rule_id: str
    name: str
    sensor_type: SensorType
    enter_threshold: float
    clear_threshold: float
    severity: RuleSeverity = RuleSeverity.WARNING
    mode: RuleMode = RuleMode.LOG_ONLY
    sustained_for_seconds: float = 0.0
    clear_delay_seconds: float = 0.0
    conditions: List[RuleCondition] = field(default_factory=list)
    staleness_policy: StalenessPolicy = StalenessPolicy.ALERT_STALE
    stale_age_seconds: int = 60
    context_gates: List[RuleCondition] = field(default_factory=list)
    enabled: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id must be non-empty")
        if not self.name:
            raise ValueError("name must be non-empty")
        if isinstance(self.sensor_type, str):
            self.sensor_type = SensorType.from_string(self.sensor_type)
        if isinstance(self.severity, str):
            self.severity = RuleSeverity(self.severity)
        if isinstance(self.mode, str):
            self.mode = RuleMode(self.mode)
        if isinstance(self.staleness_policy, str):
            self.staleness_policy = StalenessPolicy(self.staleness_policy)
        self.conditions = [
            c if isinstance(c, RuleCondition) else RuleCondition.from_dict(c)
            for c in self.conditions
        ]
        self.context_gates = [
            c if isinstance(c, RuleCondition) else RuleCondition.from_dict(c)
            for c in self.context_gates
        ]

        if self.clear_threshold >= self.enter_threshold:
            raise ValueError("clear_threshold must be below enter_threshold")
        if self.sustained_for_seconds < 0:
            raise ValueError("sustained_for_seconds cannot be negative")
        if self.clear_delay_seconds < 0:
            raise ValueError("clear_delay_seconds cannot be negative")
        if self.stale_age_seconds < 1:
            raise ValueError("stale_age_seconds must be at least 1")
        _validate_confidence_metadata(self.metadata)

        primary_condition = RuleCondition(
            sensor_type=self.sensor_type,
            operator=ComparisonOperator.GT,
            threshold=self.enter_threshold,
            required=True,
            role=EvidenceRole.CORROBORATES,
            weight=float(self.metadata.get("primary_weight", 0.0)),
        )
        self.conditions = [primary_condition, *self.conditions]

    @property
    def is_cross_sensor(self) -> bool:
        sensor_keys = {
            (condition.sensor_type, condition.sensor_id)
            for condition in self.conditions
        }
        return len(sensor_keys) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "sensor_type": self.sensor_type.value,
            "enter_threshold": self.enter_threshold,
            "clear_threshold": self.clear_threshold,
            "severity": self.severity.value,
            "mode": self.mode.value,
            "sustained_for_seconds": self.sustained_for_seconds,
            "clear_delay_seconds": self.clear_delay_seconds,
            "conditions": [
                condition.to_dict()
                for condition in self.conditions[1:]
            ],
            "staleness_policy": self.staleness_policy.value,
            "stale_age_seconds": self.stale_age_seconds,
            "context_gates": [
                condition.to_dict()
                for condition in self.context_gates
            ],
            "enabled": self.enabled,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        return cls(**data)


@dataclass
class BreachState:
    """Current state for one rule/sensor key."""

    rule_id: str
    sensor_ids: List[str]
    status: BreachStatus = BreachStatus.IDLE
    first_triggered_at: Optional[datetime] = None
    last_triggered_at: Optional[datetime] = None
    clear_started_at: Optional[datetime] = None
    cleared_at: Optional[datetime] = None
    rule_snapshot: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=_utc_now_naive)

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id must be non-empty")
        if not self.sensor_ids:
            raise ValueError("sensor_ids must contain at least one sensor")
        if isinstance(self.status, str):
            self.status = BreachStatus(self.status)

    @property
    def state_key(self) -> str:
        return f"{self.rule_id}:{','.join(sorted(self.sensor_ids))}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "sensor_ids": self.sensor_ids,
            "status": self.status.value,
            "first_triggered_at": _dt_to_str(self.first_triggered_at),
            "last_triggered_at": _dt_to_str(self.last_triggered_at),
            "clear_started_at": _dt_to_str(self.clear_started_at),
            "cleared_at": _dt_to_str(self.cleared_at),
            "rule_snapshot": self.rule_snapshot,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class BreachLogEntry:
    """Append-only audit entry for an official breach."""

    rule_id: str
    sensor_ids: List[str]
    triggered_at: datetime
    context_snapshot: Dict[str, Any]
    breach_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cleared_at: Optional[datetime] = None
    severity: RuleSeverity = RuleSeverity.WARNING
    escalated_to_tier3: bool = False
    tier3_decision: Optional[Dict[str, Any]] = None
    action_taken: Optional[Dict[str, Any]] = None
    human_reviewed: bool = False
    human_notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id must be non-empty")
        if not self.sensor_ids:
            raise ValueError("sensor_ids must contain at least one sensor")
        if isinstance(self.severity, str):
            self.severity = RuleSeverity(self.severity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breach_id": self.breach_id,
            "rule_id": self.rule_id,
            "sensor_ids": self.sensor_ids,
            "triggered_at": self.triggered_at.isoformat(),
            "cleared_at": _dt_to_str(self.cleared_at),
            "severity": self.severity.value,
            "context_snapshot": self.context_snapshot,
            "escalated_to_tier3": self.escalated_to_tier3,
            "tier3_decision": self.tier3_decision,
            "action_taken": self.action_taken,
            "human_reviewed": self.human_reviewed,
            "human_notes": self.human_notes,
        }


def _dt_to_str(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _validate_confidence_metadata(metadata: Dict[str, Any]) -> None:
    settings = {
        "base_confidence": metadata.get("base_confidence", DEFAULT_BASE_CONFIDENCE),
        "max_confidence": metadata.get("max_confidence", DEFAULT_MAX_CONFIDENCE),
        "missing_required_penalty": metadata.get(
            "missing_required_penalty",
            DEFAULT_MISSING_REQUIRED_PENALTY,
        ),
        "stale_required_penalty": metadata.get(
            "stale_required_penalty",
            DEFAULT_STALE_REQUIRED_PENALTY,
        ),
        "primary_weight": metadata.get("primary_weight", 0.0),
    }
    for name, value in settings.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must be between 0.0 and 1.0")
    if float(settings["base_confidence"]) > float(settings["max_confidence"]):
        raise ValueError("base_confidence cannot exceed max_confidence")


__all__ = [
    "DEFAULT_CROSS_SENSOR_HISTORY_SECONDS",
    "DEFAULT_CLUSTER_WINDOW_MS",
    "DEFAULT_BASE_CONFIDENCE",
    "DEFAULT_MAX_CONFIDENCE",
    "DEFAULT_MISSING_REQUIRED_PENALTY",
    "DEFAULT_STALE_REQUIRED_PENALTY",
    "RuleSeverity",
    "RuleMode",
    "StalenessPolicy",
    "BreachStatus",
    "ComparisonOperator",
    "ContextEffect",
    "EvidenceRole",
    "RuleCondition",
    "Rule",
    "BreachState",
    "BreachLogEntry",
]
