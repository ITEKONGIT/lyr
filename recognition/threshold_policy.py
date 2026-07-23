"""
Staleness policy engine for Tier 2.7 evidence snapshots.

This module decides what to do with missing or stale evidence. It does not
calculate confidence or update breach state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from .threshold_contracts import Rule, StalenessPolicy
from .threshold_evidence import EvidenceItem, EvidenceSnapshot, EvidenceStatus


class PolicyAction(str, Enum):
    CONTINUE = "continue"
    SUPPRESS = "suppress"
    STALE_ALERT = "stale_alert"


@dataclass
class PolicyIssue:
    """One missing or stale evidence item that affected policy."""

    condition_index: int
    sensor_type: str
    status: EvidenceStatus
    required: bool
    sensor_id: str | None = None
    explanation: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = EvidenceStatus(self.status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_index": self.condition_index,
            "sensor_type": self.sensor_type,
            "sensor_id": self.sensor_id,
            "status": self.status.value,
            "required": self.required,
            "explanation": self.explanation,
        }


@dataclass
class PolicyDecision:
    """Deterministic staleness policy decision for one evidence snapshot."""

    rule_id: str
    action: PolicyAction
    staleness_policy: StalenessPolicy
    issues: List[PolicyIssue] = field(default_factory=list)
    explanation: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.action, str):
            self.action = PolicyAction(self.action)
        if isinstance(self.staleness_policy, str):
            self.staleness_policy = StalenessPolicy(self.staleness_policy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "action": self.action.value,
            "staleness_policy": self.staleness_policy.value,
            "issues": [issue.to_dict() for issue in self.issues],
            "explanation": self.explanation,
        }


def apply_staleness_policy(
    rule: Rule,
    snapshot: EvidenceSnapshot,
) -> PolicyDecision:
    """Apply rule staleness policy to missing/stale evidence."""
    issues = [
        _issue_from_item(item)
        for item in snapshot.items
        if item.status in (EvidenceStatus.MISSING, EvidenceStatus.STALE)
    ]
    required_issues = [issue for issue in issues if issue.required]

    if not required_issues:
        return PolicyDecision(
            rule_id=rule.rule_id,
            action=PolicyAction.CONTINUE,
            staleness_policy=rule.staleness_policy,
            issues=issues,
            explanation="No required evidence is missing or stale",
        )

    if rule.staleness_policy == StalenessPolicy.ALERT_STALE:
        return PolicyDecision(
            rule_id=rule.rule_id,
            action=PolicyAction.STALE_ALERT,
            staleness_policy=rule.staleness_policy,
            issues=required_issues,
            explanation=(
                "Required evidence is missing or stale; emit a human-review stale alert"
            ),
        )

    if rule.staleness_policy == StalenessPolicy.FAIL_CLOSED:
        return PolicyDecision(
            rule_id=rule.rule_id,
            action=PolicyAction.SUPPRESS,
            staleness_policy=rule.staleness_policy,
            issues=required_issues,
            explanation=(
                "Required evidence is missing or stale; fail_closed suppresses evaluation"
            ),
        )

    return PolicyDecision(
        rule_id=rule.rule_id,
        action=PolicyAction.CONTINUE,
        staleness_policy=rule.staleness_policy,
        issues=required_issues,
        explanation=(
            "Required evidence is missing or stale; fail_open continues with available data"
        ),
    )


def _issue_from_item(item: EvidenceItem) -> PolicyIssue:
    if item.status == EvidenceStatus.MISSING:
        detail = "missing"
    else:
        detail = f"stale after {item.age_seconds:.1f}s"

    label = item.sensor_id or item.sensor_type
    return PolicyIssue(
        condition_index=item.condition_index,
        sensor_type=item.sensor_type,
        sensor_id=item.sensor_id,
        status=item.status,
        required=item.required,
        explanation=f"{label} is {detail}",
    )


__all__ = [
    "PolicyAction",
    "PolicyDecision",
    "PolicyIssue",
    "apply_staleness_policy",
]
