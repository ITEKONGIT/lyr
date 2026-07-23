from datetime import datetime

from recognition.threshold_contracts import Rule, StalenessPolicy
from recognition.threshold_evidence import EvidenceItem, EvidenceSnapshot, EvidenceStatus
from recognition.threshold_policy import PolicyAction, apply_staleness_policy


BASE_TIME = datetime(2025, 1, 1, 12, 0, 0)


def _rule(policy=StalenessPolicy.ALERT_STALE):
    return Rule(
        rule_id="fire_detection",
        name="Potential fire detection",
        sensor_type="temperature",
        enter_threshold=36.0,
        clear_threshold=34.0,
        staleness_policy=policy,
    )


def _snapshot(*items):
    return EvidenceSnapshot(
        rule_id="fire_detection",
        triggering_sensor_id="temp_1",
        triggering_reading_id="reading_1",
        evaluated_at=BASE_TIME,
        items=list(items),
    )


def _item(status, required=True, sensor_id="humidity_1", sensor_type="humidity"):
    return EvidenceItem(
        condition_index=1,
        sensor_type=sensor_type,
        sensor_id=sensor_id,
        operator="<",
        threshold=40.0,
        status=status,
        required=required,
        age_seconds=120.0 if status == EvidenceStatus.STALE else None,
    )


def test_alert_stale_required_stale_evidence_creates_stale_alert():
    decision = apply_staleness_policy(
        _rule(StalenessPolicy.ALERT_STALE),
        _snapshot(_item(EvidenceStatus.STALE, required=True)),
    )

    assert decision.action == PolicyAction.STALE_ALERT
    assert decision.issues[0].sensor_id == "humidity_1"
    assert decision.issues[0].status == EvidenceStatus.STALE
    assert "human-review stale alert" in decision.explanation


def test_alert_stale_required_missing_evidence_creates_stale_alert():
    decision = apply_staleness_policy(
        _rule(StalenessPolicy.ALERT_STALE),
        _snapshot(_item(EvidenceStatus.MISSING, required=True)),
    )

    assert decision.action == PolicyAction.STALE_ALERT
    assert decision.issues[0].status == EvidenceStatus.MISSING
    assert decision.issues[0].explanation == "humidity_1 is missing"


def test_fail_closed_required_stale_evidence_suppresses_evaluation():
    decision = apply_staleness_policy(
        _rule(StalenessPolicy.FAIL_CLOSED),
        _snapshot(_item(EvidenceStatus.STALE, required=True)),
    )

    assert decision.action == PolicyAction.SUPPRESS
    assert "fail_closed suppresses" in decision.explanation


def test_fail_closed_required_missing_evidence_suppresses_evaluation():
    decision = apply_staleness_policy(
        _rule(StalenessPolicy.FAIL_CLOSED),
        _snapshot(_item(EvidenceStatus.MISSING, required=True)),
    )

    assert decision.action == PolicyAction.SUPPRESS
    assert decision.issues[0].required is True


def test_fail_open_required_stale_evidence_continues_with_warning_metadata():
    decision = apply_staleness_policy(
        _rule(StalenessPolicy.FAIL_OPEN),
        _snapshot(_item(EvidenceStatus.STALE, required=True)),
    )

    assert decision.action == PolicyAction.CONTINUE
    assert decision.issues[0].status == EvidenceStatus.STALE
    assert "fail_open continues" in decision.explanation


def test_optional_missing_or_stale_evidence_never_suppresses():
    decision = apply_staleness_policy(
        _rule(StalenessPolicy.FAIL_CLOSED),
        _snapshot(
            _item(EvidenceStatus.MISSING, required=False, sensor_id="smoke_1"),
            _item(EvidenceStatus.STALE, required=False, sensor_id="gas_1"),
        ),
    )

    assert decision.action == PolicyAction.CONTINUE
    assert len(decision.issues) == 2
    assert all(issue.required is False for issue in decision.issues)
    assert "No required evidence" in decision.explanation


def test_matched_and_not_matched_evidence_continue_without_issues():
    decision = apply_staleness_policy(
        _rule(StalenessPolicy.ALERT_STALE),
        _snapshot(
            _item(EvidenceStatus.MATCHED, required=True),
            _item(EvidenceStatus.NOT_MATCHED, required=True, sensor_id="smoke_1"),
        ),
    )

    assert decision.action == PolicyAction.CONTINUE
    assert decision.issues == []
    assert decision.to_dict()["action"] == "continue"


def test_decision_metadata_names_all_required_problem_evidence():
    decision = apply_staleness_policy(
        _rule(StalenessPolicy.ALERT_STALE),
        _snapshot(
            _item(EvidenceStatus.MISSING, required=True, sensor_id="humidity_1"),
            _item(EvidenceStatus.STALE, required=True, sensor_id="smoke_1", sensor_type="smoke"),
        ),
    )

    data = decision.to_dict()

    assert data["action"] == "stale_alert"
    assert data["staleness_policy"] == "alert_stale"
    assert [issue["sensor_id"] for issue in data["issues"]] == [
        "humidity_1",
        "smoke_1",
    ]
    assert data["issues"][1]["explanation"] == "smoke_1 is stale after 120.0s"
