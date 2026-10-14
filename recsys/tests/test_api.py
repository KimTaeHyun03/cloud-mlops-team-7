r"""
API 테스트 (정상 200 / 오류 422 / 미지 유저·영화 fallback)
파일 위치: D:\ysu_26_2\MLOPS\recsys\tests\test_api.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  pytest recsys/tests
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recsys.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_version"]


def test_predict_ok():
    r = client.post("/predict", json={"user_id": 1, "movie_id": 318})
    assert r.status_code == 200
    body = r.json()
    assert 0.5 <= body["predicted_rating"] <= 5.0
    assert body["fallback"] is False
    assert body["title"].startswith("Shawshank")


def test_predict_unknown_user_fallback():
    r = client.post("/predict", json={"user_id": 99999, "movie_id": 318})
    assert r.status_code == 200
    body = r.json()
    assert body["fallback"] is True
    assert 0.5 <= body["predicted_rating"] <= 5.0


def test_predict_unknown_movie_404():
    r = client.post("/predict", json={"user_id": 1, "movie_id": 99999999})
    assert r.status_code == 404


@pytest.mark.parametrize("payload", [
    {"user_id": "abc", "movie_id": 318},
    {"user_id": 0, "movie_id": 318},
    {"user_id": -5, "movie_id": 318},
    {"user_id": 1},
    {"user_id": 1, "movie_id": 1.5},
])
def test_predict_invalid_422(payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_recommend_ok_and_excludes_seen():
    r = client.post("/recommend", json={"user_id": 1, "k": 10})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 10
    scores = [i["predicted_rating"] for i in items]
    assert scores == sorted(scores, reverse=True)
    seen = {h["movieId"] for h in client.get("/users/1/history?limit=100").json()}
    assert not seen & {i["movieId"] for i in items}


def test_recommend_unknown_user_fallback():
    r = client.post("/recommend", json={"user_id": 99999, "k": 5})
    assert r.status_code == 200
    assert r.json()["fallback"] is True
    assert len(r.json()["items"]) == 5


@pytest.mark.parametrize("k", [0, 51, 100, "x"])
def test_recommend_invalid_k_422(k):
    assert client.post("/recommend", json={"user_id": 1, "k": k}).status_code == 422


def test_helpers_and_index():
    assert len(client.get("/users").json()) == 610
    assert client.get("/movies?q=toy story").json()
    assert len(client.get("/samples?n=5").json()) == 5
    assert client.get("/").status_code == 200
    assert client.get("/users/abc/history").status_code == 422
