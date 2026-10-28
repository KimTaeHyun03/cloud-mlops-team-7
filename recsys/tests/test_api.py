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


# ---------- 내 평점으로 추천 (fold-in) ----------
MY = [{"movie_id": 1, "rating": 5.0}, {"movie_id": 318, "rating": 4.5},
      {"movie_id": 296, "rating": 4.0}, {"movie_id": 2571, "rating": 5.0}]


def test_from_ratings_ok():
    r = client.post("/recommend/from-ratings", json={"ratings": MY, "k": 10})
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 10
    assert not {m["movie_id"] for m in MY} & {i["movieId"] for i in body["items"]}
    scores = [i["predicted_rating"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert len(body["similar_users"]) == 5
    assert body["n_ratings_used"] == 4


def test_from_ratings_deterministic():
    a = client.post("/recommend/from-ratings", json={"ratings": MY}).json()
    b = client.post("/recommend/from-ratings", json={"ratings": MY}).json()
    assert [i["movieId"] for i in a["items"]] == [i["movieId"] for i in b["items"]]


def test_from_ratings_changes_with_taste():
    low = [{**m, "rating": 0.5} for m in MY]
    a = client.post("/recommend/from-ratings", json={"ratings": MY}).json()["items"]
    b = client.post("/recommend/from-ratings", json={"ratings": low}).json()["items"]
    assert [i["movieId"] for i in a] != [i["movieId"] for i in b]


@pytest.mark.parametrize("payload", [
    {"ratings": MY[:2]},                                              # 3개 미만
    {"ratings": MY + [{"movie_id": 1, "rating": 3.0}]},               # 중복
    {"ratings": [{"movie_id": 1, "rating": 6.0}] + MY[1:]},           # 범위 초과
    {"ratings": [{"movie_id": 1, "rating": 3.3}] + MY[1:]},           # 0.5 단위 아님
    {"ratings": [{"movie_id": 99999999, "rating": 3.0}] + MY[1:]},    # 없는 영화
    {"ratings": MY, "k": 0},
])
def test_from_ratings_invalid_422(payload):
    assert client.post("/recommend/from-ratings", json=payload).status_code == 422


def test_popular():
    items = client.get("/popular?limit=5").json()
    assert len(items) == 5 and items[0]["n_ratings"] >= items[-1]["n_ratings"]


def _last_log(tmp_path, monkeypatch):
    import json
    from recsys import config
    return json.loads(config.REQUEST_LOG_PATH.read_text(encoding="utf-8").splitlines()[-1])


def test_request_id_matches_log(tmp_path, monkeypatch):
    r = client.post("/predict", json={"user_id": 1, "movie_id": 318})
    log = _last_log(tmp_path, monkeypatch)
    assert r.json()["request_id"] == r.headers["x-request-id"] == log["request_id"]
    assert log["status"] == "ok" and log["model_version"] == r.json()["model_version"]
    assert log["input"] == {"user_id": 1, "movie_id": 318}


def test_error_request_is_logged_with_header_id(tmp_path, monkeypatch):
    r = client.post("/predict", json={"user_id": "abc", "movie_id": 1})
    log = _last_log(tmp_path, monkeypatch)
    assert r.status_code == 422
    assert log["request_id"] == r.headers["x-request-id"] and log["error_type"] == "validation_error"
