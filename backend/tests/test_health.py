def test_health_svarer_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    # Uten leverandoer-secrets skal mailer falle tilbake til ren logging.
    assert body["mailer"] == "log"


def test_root(client):
    assert client.get("/").json()["service"] == "bestefar-api"
