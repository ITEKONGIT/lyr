from fastapi import FastAPI
from fastapi.testclient import TestClient

from recognition.api import create_app
from recognition.api.router import router
from recognition.config import Config
from recognition.sensor_history import MAX_QUERY_LIMIT, reset_store_for_testing
from recognition.sensor_registry import reset_registry


def _client(tmp_path):
    reset_registry()
    reset_store_for_testing(tmp_path / "history.db")
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _headers():
    return {"X-API-Key": Config.API_KEY}


def test_single_ingest_accepts_aware_timestamp(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/api/v1/sensors/ingest",
        headers=_headers(),
        json={
            "sensor_id": "temp_001",
            "sensor_type": "temperature",
            "value": 22.5,
            "timestamp": "2026-07-23T04:00:00Z",
            "unit": "°C",
            "confidence_score": 0.9,
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "accepted"


def test_http_sensor_ingest_requires_api_key(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/api/v1/sensors/ingest",
        json={
            "sensor_id": "temp_001",
            "sensor_type": "temperature",
            "value": 22.5,
        },
    )

    assert response.status_code == 401


def test_batch_ingest_returns_partial_item_statuses(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/api/v1/sensors/ingest/batch",
        headers=_headers(),
        json={
            "readings": [
                {
                    "sensor_id": "temp_001",
                    "sensor_type": "temperature",
                    "value": 22.5,
                    "confidence_score": 0.9,
                },
                {
                    "sensor_id": "",
                    "sensor_type": "temperature",
                    "value": 23.0,
                    "confidence_score": 0.9,
                },
            ]
        },
    )

    data = response.json()

    assert response.status_code == 200
    assert data["status"] == "partial_success"
    assert data["accepted"] == 1
    assert data["failed"] == 1
    assert data["results"][0]["status"] == "accepted"
    assert data["results"][1]["status"] == "rejected"


def test_unknown_sensor_status_and_history_are_404(tmp_path):
    client = _client(tmp_path)

    status_response = client.get(
        "/api/v1/sensors/status/missing_sensor",
        headers=_headers(),
    )
    history_response = client.get(
        "/api/v1/sensors/history/missing_sensor",
        headers=_headers(),
    )

    assert status_response.status_code == 404
    assert history_response.status_code == 404


def test_face_history_redacts_identity_metadata(tmp_path):
    client = _client(tmp_path)

    ingest = client.post(
        "/api/v1/sensors/ingest",
        headers=_headers(),
        json={
            "sensor_id": "face_camera_1",
            "sensor_type": "face",
            "value": 1.0,
            "confidence_score": 0.99,
            "metadata": {
                "identity": "Known: Ada",
                "name": "Ada",
                "user_id": "ada_001",
                "is_live": True,
            },
        },
    )
    history = client.get(
        "/api/v1/sensors/history/face_camera_1",
        headers=_headers(),
    )

    reading = history.json()["readings"][0]

    assert ingest.status_code == 201
    assert history.status_code == 200
    assert reading["metadata_redacted"] is True
    assert reading["metadata"] == {"is_live": True}


def test_sensor_query_rejects_like_and_huge_limits(tmp_path):
    client = _client(tmp_path)

    like_response = client.post(
        "/api/v1/sensors/query",
        headers=_headers(),
        json={
            "filters": [
                {"field": "sensor_id", "op": "LIKE", "value": "%temp%"}
            ],
            "limit": 10,
        },
    )
    limit_response = client.post(
        "/api/v1/sensors/query",
        headers=_headers(),
        json={"filters": [], "limit": MAX_QUERY_LIMIT + 1},
    )

    assert like_response.status_code == 403
    assert limit_response.status_code == 422


def test_create_app_rejects_oversized_sensor_body_before_parsing():
    client = TestClient(create_app())
    huge_metadata = "x" * 1_000_001

    response = client.post(
        "/api/v1/sensors/ingest",
        headers=_headers(),
        json={
            "sensor_id": "temp_oversized",
            "sensor_type": "temperature",
            "value": 22.0,
            "metadata": {"payload": huge_metadata},
        },
    )

    assert response.status_code == 413


def test_websocket_ingest_acknowledges_valid_reading(tmp_path):
    client = _client(tmp_path)

    with client.websocket_connect(
        f"/api/v1/sensors/ingest/ws/temp_ws?api_key={Config.API_KEY}"
    ) as websocket:
        websocket.send_json({
            "sensor_type": "temperature",
            "value": 21.5,
            "confidence_score": 0.9,
        })
        response = websocket.receive_json()

    assert response["status"] == "accepted"
    assert response["reading_id"]


def test_websocket_ingest_rejects_sensor_id_mismatch(tmp_path):
    client = _client(tmp_path)

    with client.websocket_connect(
        f"/api/v1/sensors/ingest/ws/temp_ws?api_key={Config.API_KEY}"
    ) as websocket:
        websocket.send_json({
            "sensor_id": "other_sensor",
            "sensor_type": "temperature",
            "value": 21.5,
            "confidence_score": 0.9,
        })
        response = websocket.receive_json()

    assert response["status"] == "rejected"
    assert "does not match" in response["error"]


def test_websocket_ingest_rejects_missing_api_key(tmp_path):
    client = _client(tmp_path)

    with client.websocket_connect("/api/v1/sensors/ingest/ws/temp_ws") as websocket:
        response = websocket.receive_json()

    assert response["status"] == "rejected"
    assert "Invalid or missing API key" in response["error"]
