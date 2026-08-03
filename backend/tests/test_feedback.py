MELDING = {"subject": "Scan feiler i motlys",
           "body": "Appen klarer ikke aa lese skjermen naar sola staar lavt.",
           "app_version": "0.13", "device_model": "Pixel 7"}


def test_feedback_lagres(client):
    r = client.post("/v1/feedback", json=MELDING)
    assert r.status_code == 202
    assert r.json()["status"] == "mottatt"

    from app.db import db
    from app.models import Feedback

    s = next(db())
    try:
        row = s.get(Feedback, r.json()["id"])
        assert row.subject == MELDING["subject"]
        assert row.app_version == "0.13"
        # Uten FEEDBACK_TO er videresending en no-op, men skal telle som utfoert.
        assert row.forward_error is None
        assert row.forwarded_at is not None
    finally:
        s.close()


def test_feedback_avviser_tom_melding(client):
    assert client.post("/v1/feedback", json={"subject": "", "body": ""}).status_code == 422


def test_feedback_rate_limit(client):
    # FEEDBACK_RATE_PER_HOUR er 5 som standard.
    for _ in range(5):
        assert client.post("/v1/feedback", json=MELDING).status_code == 202
    assert client.post("/v1/feedback", json=MELDING).status_code == 429
