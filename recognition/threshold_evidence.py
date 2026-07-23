"""
Evidence snapshot builder for Tier 2.7 multi-evidence rules.

This module gathers and classifies rule evidence only. It does not apply
staleness policies, calculate confidence, or update breach state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .sensor_contracts import SensorReading
from .sensor_history import HistoryStore
from .threshold_contracts import (
    ComparisonOperator,
    EvidenceRole,
    Rule,
    RuleCondition,
    _utc_now_naive,
)


class EvidenceStatus(str, Enum):
    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    MISSING = "missing"
    STALE = "stale"


@dataclass
class EvidenceItem:
    """One condition and the reading used to evaluate it."""

    condition_index: int
    sensor_type: str
    operator: str
    threshold: float
    status: EvidenceStatus
    sensor_id: Optional[str] = None
    required: bool = True
    role: EvidenceRole = EvidenceRole.CORROBORATES
    weight: float = 0.0
    history_window_seconds: int = 0
    reason: str = ""
    reading: Optional[Dict[str, Any]] = None
    age_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = EvidenceStatus(self.status)
        if isinstance(self.role, str):
            self.role = EvidenceRole(self.role)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_index": self.condition_index,
            "sensor_type": self.sensor_type,
            "sensor_id": self.sensor_id,
            "operator": self.operator,
            "threshold": self.threshold,
            "required": self.required,
            "role": self.role.value,
            "weight": self.weight,
            "history_window_seconds": self.history_window_seconds,
            "status": self.status.value,
            "reason": self.reason,
            "reading": self.reading,
            "age_seconds": self.age_seconds,
        }


@dataclass
class EvidenceSnapshot:
    """All currently available evidence for a rule evaluation."""

    rule_id: str
    triggering_sensor_id: str
    triggering_reading_id: str
    evaluated_at: datetime
    items: List[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "triggering_sensor_id": self.triggering_sensor_id,
            "triggering_reading_id": self.triggering_reading_id,
            "evaluated_at": self.evaluated_at.isoformat(),
            "items": [item.to_dict() for item in self.items],
        }


def build_evidence_snapshot(
    reading: SensorReading,
    rule: Rule,
    history_store: Optional[HistoryStore] = None,
    now: Optional[datetime] = None,
) -> EvidenceSnapshot:
    """Build a point-in-time evidence snapshot for one rule and reading."""
    evaluated_at = now or _utc_now_naive()
    items = [
        _evaluate_condition(
            condition=condition,
            condition_index=index,
            current_reading=reading,
            history_store=history_store,
            now=evaluated_at,
            use_current=index == 0,
        )
        for index, condition in enumerate(rule.conditions)
    ]
    return EvidenceSnapshot(
        rule_id=rule.rule_id,
        triggering_sensor_id=reading.sensor_id,
        triggering_reading_id=reading.reading_id,
        evaluated_at=evaluated_at,
        items=items,
    )


def _evaluate_condition(
    condition: RuleCondition,
    condition_index: int,
    current_reading: SensorReading,
    history_store: Optional[HistoryStore],
    now: datetime,
    use_current: bool,
) -> EvidenceItem:
    reading = current_reading if use_current else _latest_reading(condition, history_store)
    base = {
        "condition_index": condition_index,
        "sensor_type": condition.sensor_type.value,
        "sensor_id": condition.sensor_id,
        "operator": condition.operator.value,
        "threshold": condition.threshold,
        "required": condition.required,
        "role": condition.role,
        "weight": condition.weight,
        "history_window_seconds": condition.history_window_seconds,
    }

    if reading is None:
        return EvidenceItem(
            **base,
            status=EvidenceStatus.MISSING,
            reason=condition.reason or "No reading available for condition",
        )

    age_seconds = max(0.0, (now - reading.timestamp).total_seconds())
    item_reading = _reading_to_dict(reading)
    if age_seconds > condition.history_window_seconds:
        return EvidenceItem(
            **base,
            status=EvidenceStatus.STALE,
            reading=item_reading,
            age_seconds=age_seconds,
            reason=condition.reason or "Reading is outside condition history window",
        )

    if _matches_condition(reading, condition):
        return EvidenceItem(
            **base,
            status=EvidenceStatus.MATCHED,
            reading=item_reading,
            age_seconds=age_seconds,
            reason=condition.reason or "Condition matched",
        )

    return EvidenceItem(
        **base,
        status=EvidenceStatus.NOT_MATCHED,
        reading=item_reading,
        age_seconds=age_seconds,
        reason=condition.reason or "Condition did not match",
    )


def _latest_reading(
    condition: RuleCondition,
    history_store: Optional[HistoryStore],
) -> Optional[SensorReading]:
    if history_store is None:
        return None
    if condition.sensor_id:
        readings = history_store.get_history(condition.sensor_id, limit=1)
    else:
        readings = history_store.get_history_by_type(condition.sensor_type, limit=1)
    return readings[0] if readings else None


def _reading_to_dict(reading: SensorReading) -> Dict[str, Any]:
    return {
        "reading_id": reading.reading_id,
        "sensor_id": reading.sensor_id,
        "sensor_type": reading.sensor_type.value,
        "value": reading.value,
        "timestamp": reading.timestamp.isoformat(),
        "confidence_score": reading.confidence_score,
        "source": reading.source,
        "location": reading.location,
    }


def _matches_condition(reading: SensorReading, condition: RuleCondition) -> bool:
    if reading.sensor_type != condition.sensor_type:
        return False
    if condition.sensor_id and reading.sensor_id != condition.sensor_id:
        return False

    comparisons = {
        ComparisonOperator.GT: reading.value > condition.threshold,
        ComparisonOperator.GTE: reading.value >= condition.threshold,
        ComparisonOperator.LT: reading.value < condition.threshold,
        ComparisonOperator.LTE: reading.value <= condition.threshold,
        ComparisonOperator.EQ: reading.value == condition.threshold,
        ComparisonOperator.NEQ: reading.value != condition.threshold,
    }
    if condition.operator not in comparisons:
        raise ValueError(
            f"Operator {condition.operator.value} is accepted by the contract "
            "but is not implemented by evidence evaluation yet"
        )
    return comparisons[condition.operator]


__all__ = [
    "EvidenceItem",
    "EvidenceSnapshot",
    "EvidenceStatus",
    "build_evidence_snapshot",
]
