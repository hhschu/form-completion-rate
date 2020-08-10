from fastapi.testclient import TestClient
from pytest import fixture, mark

import app.api
import app.main


@fixture()
def client():
    client = TestClient(app.main.app)
    return client


def test_heartbeat(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == "ok"


def test_get_infer(client):
    response = client.get("/infer")

    assert response.status_code == 200
    assert response.json() == {"versions": [1596729586]}


def test_post_infer(client):
    payload = {f"feat_{i:02}": 0 for i in range(1, 48)}
    response = client.post("/infer", json=payload, allow_redirects=True)

    assert response.status_code == 200
    assert isinstance(response.json()["score"], float)


@mark.dependency()
def test_load_model(client):
    response = client.post("/load/1596729183", json={}, allow_redirects=True)

    assert response.status_code == 200
    assert list(app.api.Models.keys()) == ["1596729183"]


@mark.dependency(depends=["test_load_model"])
def test_post_infer_with_specific_model(client):
    payload = {f"feat_{i:02}": 0 for i in range(1, 48)}
    response = client.post("/infer/1596729183", json=payload, allow_redirects=True)

    assert response.status_code == 200
    assert isinstance(response.json()["score"], float)


def test_post_infer_with_non_exist_model(client):
    payload = {f"feat_{i:02}": 0 for i in range(1, 48)}
    response = client.post("/infer/42", json=payload, allow_redirects=True)

    assert response.status_code == 404
