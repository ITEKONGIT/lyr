"""
Tier 2 single-sensor threshold gate.

Phase 2.5 evaluates one incoming SensorReading against deterministic threshold
rules and persists the breach lifecycle state. Cross-sensor aggregation,
context gates, audit logging, and Tier 3 escalation are intentionally left for
later phases.
"""

import copy
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .sensor_contracts import SensorReading
from .sensor_history import HistoryStore
from .threshold_contracts import (
    BreachState,
    BreachStatus,
    ComparisonOperator,
    Rule,
    RuleCondition,
    RuleMode,
    RuleSeverity,
    StalenessPolicy,
    _utc_now_naive,
)
from .threshold_state import BreachStateStore


SEVERITY_ORDER = [
    RuleSeverity.INFO,
    RuleSeverity.WARNING,
    RuleSeverity.HIGH,
    RuleSeverity.CRITICAL,
]


class ThresholdGate:
    """Deterministic rule evaluator for single-sensor threshold breaches."""

    def __init__(
        self,
        rules: Iterable[Rule],
        state_store: Optional[BreachStateStore] = None,
        history_store: Optional[HistoryStore] = None,
    ):
        self.rules = list(rules)
        self.state_store = state_store or BreachStateStore()
        self.history_store = history_store

    def evaluate(
        self,
        reading: SensorReading,
        now: Optional[datetime] = None,
    ) -> List[BreachState]:
        """
        Evaluate a reading against matching single-sensor rules.

        Returns the states changed or refreshed by this reading. If the reading
        is below threshold and no previous state exists, the result is empty.
        """
        evaluated_at = now or _utc_now_naive()
        states = []
        for rule in self.rules:
            if not self._rule_applies(rule, reading):
                continue
            effective_rule, context, suppressed = self._apply_context(
                rule,
                evaluated_at,
            )
            if suppressed:
                continue
            state = self._evaluate_rule(effective_rule, reading, evaluated_at)
            if state is not None:
                states.append(state)
        return states

    def _rule_applies(self, rule: Rule, reading: SensorReading) -> bool:
        return (
            rule.enabled
            and not rule.is_cross_sensor
            and rule.sensor_type == reading.sensor_type
        )

    def _evaluate_rule(
        self,
        rule: Rule,
        reading: SensorReading,
        now: datetime,
    ) -> Optional[BreachState]:
        sensor_ids = [reading.sensor_id]
        previous = self.state_store.get(rule.rule_id, sensor_ids)

        if previous is None:
            if not self._entered(reading, rule):
                return None
            if rule.sustained_for_seconds == 0:
                return self._save_active(rule, sensor_ids, now, now)
            return self._save_pending(rule, sensor_ids, now, now)

        status = previous.status
        if status in (BreachStatus.IDLE, BreachStatus.CLEARED):
            return self._from_idle_or_cleared(previous, rule, reading, now)
        if status == BreachStatus.PENDING_SUSTAIN:
            return self._from_pending(previous, rule, reading, now)
        if status == BreachStatus.ACTIVE:
            return self._from_active(previous, rule, reading, now)
        if status == BreachStatus.CLEARING:
            return self._from_clearing(previous, rule, reading, now)
        return previous

    def _apply_context(
        self,
        rule: Rule,
        now: datetime,
    ) -> Tuple[Rule, Dict[str, Any], bool]:
        effects = rule.metadata.get("context_effects", [])
        context = {
            "evaluated_at": now.isoformat(),
            "effects": [],
            "suppressed": False,
            "ai_context": rule.metadata.get("ai_context"),
        }
        if not effects:
            return rule, context, False

        effective_rule = copy.copy(rule)
        effective_rule.metadata = copy.deepcopy(rule.metadata)
        severity = rule.severity
        mode = rule.mode
        suppressed = False

        for effect in effects:
            decision = self._evaluate_context_effect(effect, rule, now)
            context["effects"].append(decision)

            if decision["status"] in ("missing", "stale"):
                if decision["required"] and rule.staleness_policy == StalenessPolicy.FAIL_CLOSED:
                    suppressed = True
                    decision["reason"] = (
                        "Required context unavailable; fail_closed suppressed breach evaluation"
                    )
                continue

            if decision["status"] != "matched":
                continue

            severity = _adjust_severity(severity, int(effect.get("severity_delta", 0)))
            if "mode" in effect:
                mode = RuleMode(effect["mode"])
            decision["applied_severity"] = severity.value
            decision["applied_mode"] = mode.value

        context["suppressed"] = suppressed
        effective_rule.severity = severity
        effective_rule.mode = mode
        effective_rule.metadata["context_evaluation"] = context
        return effective_rule, context, suppressed

    def _evaluate_context_effect(
        self,
        effect: Dict[str, Any],
        rule: Rule,
        now: datetime,
    ) -> Dict[str, Any]:
        condition_data = effect.get("condition") or effect.get("when")
        if not condition_data:
            raise ValueError("context effect must include a condition")

        condition = RuleCondition.from_dict(condition_data)
        required = bool(effect.get("required", condition.required))
        reading = self._latest_context_reading(condition)
        decision = {
            "name": effect.get("name", "context_effect"),
            "sensor_type": condition.sensor_type.value,
            "sensor_id": condition.sensor_id,
            "operator": condition.operator.value,
            "threshold": condition.threshold,
            "required": required,
            "status": "missing",
            "reason": effect.get("reason", "Context reading unavailable"),
            "reading": None,
        }

        if reading is None:
            return decision

        decision["reading"] = {
            "sensor_id": reading.sensor_id,
            "sensor_type": reading.sensor_type.value,
            "value": reading.value,
            "timestamp": reading.timestamp.isoformat(),
            "confidence_score": reading.confidence_score,
            "source": reading.source,
        }

        max_age = condition.history_window_seconds or rule.stale_age_seconds
        if (now - reading.timestamp).total_seconds() > max_age:
            decision["status"] = "stale"
            decision["reason"] = "Context reading is stale"
            return decision

        if _matches_condition(reading, condition):
            decision["status"] = "matched"
            decision["reason"] = effect.get("reason", "Context condition matched")
            return decision

        decision["status"] = "not_matched"
        decision["reason"] = effect.get("not_matched_reason", "Context condition did not match")
        return decision

    def _latest_context_reading(
        self,
        condition: RuleCondition,
    ) -> Optional[SensorReading]:
        if self.history_store is None:
            return None

        if condition.sensor_id:
            readings = self.history_store.get_history(condition.sensor_id, limit=1)
        else:
            readings = self.history_store.get_history_by_type(
                condition.sensor_type,
                limit=1,
            )
        return readings[0] if readings else None

    def _from_idle_or_cleared(
        self,
        previous: BreachState,
        rule: Rule,
        reading: SensorReading,
        now: datetime,
    ) -> BreachState:
        if not self._entered(reading, rule):
            return self._save_idle(rule, previous.sensor_ids, now)
        if rule.sustained_for_seconds == 0:
            return self._save_active(rule, previous.sensor_ids, now, now)
        return self._save_pending(rule, previous.sensor_ids, now, now)

    def _from_pending(
        self,
        previous: BreachState,
        rule: Rule,
        reading: SensorReading,
        now: datetime,
    ) -> BreachState:
        if not self._entered(reading, rule):
            return self._save_idle(rule, previous.sensor_ids, now)

        first_triggered_at = previous.first_triggered_at or now
        elapsed = (now - first_triggered_at).total_seconds()
        if elapsed >= rule.sustained_for_seconds:
            return self._save_active(
                rule,
                previous.sensor_ids,
                first_triggered_at,
                now,
            )
        return self._save_pending(
            rule,
            previous.sensor_ids,
            first_triggered_at,
            now,
        )

    def _from_active(
        self,
        previous: BreachState,
        rule: Rule,
        reading: SensorReading,
        now: datetime,
    ) -> BreachState:
        first_triggered_at = previous.first_triggered_at or now
        if self._cleared(reading, rule):
            if rule.clear_delay_seconds == 0:
                return self._save_cleared(
                    rule,
                    previous.sensor_ids,
                    first_triggered_at,
                    previous.last_triggered_at or now,
                    now,
                    now,
                )
            return self._save_clearing(
                rule,
                previous.sensor_ids,
                first_triggered_at,
                previous.last_triggered_at or now,
                now,
                now,
            )
        return self._save_active(rule, previous.sensor_ids, first_triggered_at, now)

    def _from_clearing(
        self,
        previous: BreachState,
        rule: Rule,
        reading: SensorReading,
        now: datetime,
    ) -> BreachState:
        first_triggered_at = previous.first_triggered_at or now
        if self._entered(reading, rule):
            return self._save_active(rule, previous.sensor_ids, first_triggered_at, now)

        clear_started_at = previous.clear_started_at or now
        if self._cleared(reading, rule):
            elapsed = (now - clear_started_at).total_seconds()
            if elapsed >= rule.clear_delay_seconds:
                return self._save_cleared(
                    rule,
                    previous.sensor_ids,
                    first_triggered_at,
                    previous.last_triggered_at or now,
                    clear_started_at,
                    now,
                )

        return self._save_clearing(
            rule,
            previous.sensor_ids,
            first_triggered_at,
            previous.last_triggered_at or now,
            clear_started_at,
            now,
        )

    def _save_idle(
        self,
        rule: Rule,
        sensor_ids: List[str],
        now: datetime,
    ) -> BreachState:
        return self.state_store.upsert(
            BreachState(
                rule_id=rule.rule_id,
                sensor_ids=sensor_ids,
                status=BreachStatus.IDLE,
                rule_snapshot=rule.to_dict(),
                updated_at=now,
            )
        )

    def _save_pending(
        self,
        rule: Rule,
        sensor_ids: List[str],
        first_triggered_at: datetime,
        now: datetime,
    ) -> BreachState:
        return self.state_store.upsert(
            BreachState(
                rule_id=rule.rule_id,
                sensor_ids=sensor_ids,
                status=BreachStatus.PENDING_SUSTAIN,
                first_triggered_at=first_triggered_at,
                last_triggered_at=now,
                rule_snapshot=rule.to_dict(),
                updated_at=now,
            )
        )

    def _save_active(
        self,
        rule: Rule,
        sensor_ids: List[str],
        first_triggered_at: datetime,
        now: datetime,
    ) -> BreachState:
        return self.state_store.upsert(
            BreachState(
                rule_id=rule.rule_id,
                sensor_ids=sensor_ids,
                status=BreachStatus.ACTIVE,
                first_triggered_at=first_triggered_at,
                last_triggered_at=now,
                rule_snapshot=rule.to_dict(),
                updated_at=now,
            )
        )

    def _save_clearing(
        self,
        rule: Rule,
        sensor_ids: List[str],
        first_triggered_at: datetime,
        last_triggered_at: datetime,
        clear_started_at: datetime,
        now: datetime,
    ) -> BreachState:
        return self.state_store.upsert(
            BreachState(
                rule_id=rule.rule_id,
                sensor_ids=sensor_ids,
                status=BreachStatus.CLEARING,
                first_triggered_at=first_triggered_at,
                last_triggered_at=last_triggered_at,
                clear_started_at=clear_started_at,
                rule_snapshot=rule.to_dict(),
                updated_at=now,
            )
        )

    def _save_cleared(
        self,
        rule: Rule,
        sensor_ids: List[str],
        first_triggered_at: datetime,
        last_triggered_at: datetime,
        clear_started_at: datetime,
        now: datetime,
    ) -> BreachState:
        return self.state_store.upsert(
            BreachState(
                rule_id=rule.rule_id,
                sensor_ids=sensor_ids,
                status=BreachStatus.CLEARED,
                first_triggered_at=first_triggered_at,
                last_triggered_at=last_triggered_at,
                clear_started_at=clear_started_at,
                cleared_at=now,
                rule_snapshot=rule.to_dict(),
                updated_at=now,
            )
        )

    @staticmethod
    def _entered(reading: SensorReading, rule: Rule) -> bool:
        return reading.value > rule.enter_threshold

    @staticmethod
    def _cleared(reading: SensorReading, rule: Rule) -> bool:
        return reading.value < rule.clear_threshold


__all__ = ["ThresholdGate"]


def _adjust_severity(severity: RuleSeverity, delta: int) -> RuleSeverity:
    index = SEVERITY_ORDER.index(severity)
    adjusted = max(0, min(len(SEVERITY_ORDER) - 1, index + delta))
    return SEVERITY_ORDER[adjusted]


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
            f"Context operator {condition.operator.value} needs history trend support"
        )
    return comparisons[condition.operator]
