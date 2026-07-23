from datetime import datetime

from recognition.threshold_confidence import calculate_evidence_confidence
from recognition.threshold_contracts import EvidenceRole, Rule
from recognition.threshold_evidence import EvidenceItem, EvidenceSnapshot, EvidenceStatus


BASE_TIME = datetime(2025, 1, 1, 12, 0, 0)


def _rule(**metadata):
    return Rule(
        rule_id="multi_evidence",
        name="Multi evidence",
        sensor_type="temperature",
        enter_threshold=36.0,
        clear_threshold=34.0,
        metadata={
            "base_confidence": 0.50,
            "max_confidence": 0.95,
            "missing_required_penalty": 0.15,
            "stale_required_penalty": 0.20,
            **metadata,
        },
    )


def _snapshot(*items):
    return EvidenceSnapshot(
        rule_id="multi_evidence",
        triggering_sensor_id="temp_1",
        triggering_reading_id="reading_1",
        evaluated_at=BASE_TIME,
        items=list(items),
    )


def _item(
    status,
    role=EvidenceRole.CORROBORATES,
    weight=0.1,
    required=True,
    sensor_id="sensor_1",
):
    return EvidenceItem(
        condition_index=1,
        sensor_type="humidity",
        sensor_id=sensor_id,
        operator="<",
        threshold=40.0,
        status=status,
        required=required,
        role=role,
        weight=weight,
    )


def test_matched_corroborating_evidence_raises_confidence():
    result = calculate_evidence_confidence(
        _rule(),
        _snapshot(
            _item(EvidenceStatus.MATCHED, weight=0.2, sensor_id="temp_1"),
            _item(EvidenceStatus.MATCHED, weight=0.1, sensor_id="humidity_1"),
        ),
    )

    assert result.final_confidence == 0.8
    assert [c.amount for c in result.contributions] == [0.2, 0.1]
    assert result.contributions[0].reason == "Matched corroborating evidence"


def test_matched_contradicting_evidence_lowers_confidence():
    result = calculate_evidence_confidence(
        _rule(),
        _snapshot(
            _item(EvidenceStatus.MATCHED, weight=0.2, sensor_id="temp_1"),
            _item(
                EvidenceStatus.MATCHED,
                role=EvidenceRole.CONTRADICTS,
                weight=0.15,
                sensor_id="smoke_1",
            ),
        ),
    )

    assert result.final_confidence == 0.55
    assert result.contributions[1].amount == -0.15
    assert result.contributions[1].reason == "Matched contradicting evidence"


def test_missing_and_stale_required_evidence_apply_penalties():
    result = calculate_evidence_confidence(
        _rule(),
        _snapshot(
            _item(EvidenceStatus.MISSING, required=True, sensor_id="humidity_1"),
            _item(EvidenceStatus.STALE, required=True, sensor_id="smoke_1"),
        ),
    )

    assert result.final_confidence == 0.15
    assert [c.amount for c in result.contributions] == [-0.15, -0.20]
    assert result.contributions[0].reason == "Required evidence is missing"
    assert result.contributions[1].reason == "Required evidence is stale"


def test_optional_missing_or_stale_evidence_does_not_penalize():
    result = calculate_evidence_confidence(
        _rule(),
        _snapshot(
            _item(EvidenceStatus.MISSING, required=False, sensor_id="smoke_1"),
            _item(EvidenceStatus.STALE, required=False, sensor_id="rainfall_1"),
        ),
    )

    assert result.final_confidence == 0.5
    assert result.contributions == []


def test_confidence_never_exceeds_rule_maximum():
    result = calculate_evidence_confidence(
        _rule(max_confidence=0.75),
        _snapshot(
            _item(EvidenceStatus.MATCHED, weight=0.4, sensor_id="temp_1"),
            _item(EvidenceStatus.MATCHED, weight=0.4, sensor_id="smoke_1"),
        ),
    )

    assert result.final_confidence == 0.75
    assert result.max_confidence == 0.75


def test_confidence_never_drops_below_zero_and_serializes_explanation():
    result = calculate_evidence_confidence(
        _rule(missing_required_penalty=0.8, stale_required_penalty=0.8),
        _snapshot(
            _item(EvidenceStatus.MISSING, required=True, sensor_id="humidity_1"),
            _item(EvidenceStatus.STALE, required=True, sensor_id="smoke_1"),
        ),
    )

    data = result.to_dict()

    assert result.final_confidence == 0.0
    assert data["base_confidence"] == 0.5
    assert data["contributions"][0]["sensor_id"] == "humidity_1"
    assert data["contributions"][1]["status"] == "stale"
