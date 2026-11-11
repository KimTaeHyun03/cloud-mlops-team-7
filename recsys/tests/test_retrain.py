r"""
피드백 로그 재학습 테스트 (합성 데이터, AWS 호출 없음)
파일 위치: D:\ysu_26_2\MLOPS\recsys\tests\test_retrain.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  pytest recsys/tests/test_retrain.py
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recsys import retrain  # noqa: E402
from recsys.models import SVDBias  # noqa: E402

N_USERS, N_MOVIES = 15, 40
MOVIE_IDS = set(range(1, N_MOVIES + 1))


def _base_data():
    rng = np.random.RandomState(0)
    rows = [(u, m, float(rng.choice([2.0, 3.0, 3.5, 4.0, 4.5, 5.0])), 1000 + m)
            for u in range(1, N_USERS + 1) for m in range(1, 31)]            # 유저마다 영화 1~30 평가
    df = pd.DataFrame(rows, columns=retrain.COLUMNS)
    test = df[df.movieId > 26].reset_index(drop=True)                          # 27~30 은 test
    train = df[df.movieId <= 26].reset_index(drop=True)
    return train, test


def _ts(i: int) -> str:
    return datetime(2026, 11, 1, 0, 0, i, tzinfo=timezone.utc).isoformat(timespec="milliseconds")


def _req(rid, user, endpoint="/predict", status="ok"):
    return json.dumps({"timestamp": _ts(0), "event": "request", "request_id": rid, "endpoint": endpoint,
                       "status": status, "input": {"user_id": user} if endpoint != "/recommend/from-ratings" else {"n_ratings": 4}})


def _fb(rid, movie, rating, sec=1, user="omit"):
    ev = {"timestamp": _ts(sec), "event": "feedback", "request_id": rid, "movie_id": movie,
          "actual_rating": rating, "predicted_rating": 3.5}
    if user != "omit":
        ev["user_id"] = user
    return json.dumps(ev)


@pytest.fixture()
def env(tmp_path):
    train, test = _base_data()
    art = tmp_path / "artifacts"
    art.mkdir()
    model = SVDBias(n_factors=2, damping=5).fit(train)
    joblib.dump({"model": model, "version": "svd_bias-base", "data_hash": "h0"}, art / "model.joblib")
    (art / "metrics.json").write_text(json.dumps(
        {"model_name": "svd_bias", "params": {"n_factors": 2, "damping": 5}, "metrics": {}}), encoding="utf-8")
    return train, test, art, tmp_path / "fb" / "store.csv"


def _many_feedback(n=12):
    """유저 1~n 이 영화 31~40 중 하나씩에 평점 (원본에 없는 쌍). user_id 가 로그에 있는 새 형식."""
    return [_fb(f"{i:012x}", 30 + (i % 10) + 1, 4.0 if i % 2 else 2.5, sec=i % 50, user=i) for i in range(1, n + 1)]


# ---------- 추출 ----------
def test_extract_uses_user_id_in_feedback_or_joins_request_log():
    lines = [_req("a" * 12, 3), _fb("a" * 12, 31, 4.5),                       # 옛 로그: 요청과 조인
             _fb("b" * 12, 32, 3.0, user=5),                                  # 새 로그: user_id 가 피드백에 있음
             _req("c" * 12, 0, endpoint="/recommend/from-ratings"), _fb("c" * 12, 33, 4.0),   # guest -> 제외
             _fb("d" * 12, 34, 4.0),                                          # 요청 기록 없음 -> 제외
             "{깨진 json", ""]
    events, bad = retrain.parse_events(lines)
    df, stats = retrain.extract_feedback(events)
    assert bad == 1
    assert sorted(zip(df.userId, df.movieId, df.rating)) == [(3, 31, 4.5), (5, 32, 3.0)]
    assert stats["skip_no_user"] == 2


def test_extract_rejects_invalid_and_keeps_latest_duplicate():
    lines = [_fb("a" * 12, 31, 9.0, user=1), _fb("a" * 12, 31, 3.3, user=1), _fb("a" * 12, 31, True, user=1),
             _fb("a" * 12, 32, 2.0, sec=5, user=1), _fb("a" * 12, 32, 4.0, sec=9, user=1),   # 같은 쌍: 늦은 쪽
             _fb("a" * 12, 32, 1.0, sec=2, user=1)]                                          # 더 이른 기록은 무시
    df, stats = retrain.extract_feedback(retrain.parse_events(lines)[0])
    assert stats["skip_invalid"] == 3
    assert df.to_dict("records") == [{"userId": 1, "movieId": 32, "rating": 4.0, "timestamp": df.timestamp[0]}]
    assert stats["skip_duplicate"] == 2


def test_failed_requests_do_not_provide_user():
    events = retrain.parse_events([_req("a" * 12, 3, status="error"), _fb("a" * 12, 31, 4.0)])[0]
    df, stats = retrain.extract_feedback(events)
    assert df.empty and stats["skip_no_user"] == 1


# ---------- 걸러내기 ----------
def test_filter_drops_unknown_user_unknown_movie_and_pairs_in_base():
    train, test = _base_data()
    fb = pd.DataFrame([(1, 31, 4.0, 1),      # 사용 가능
                       (999, 31, 4.0, 1),    # 원본에 없는 유저
                       (1, 999, 4.0, 1),     # 영화 목록에 없음
                       (1, 5, 4.0, 1),       # 이미 train 에 있는 쌍
                       (1, 28, 4.0, 1)],     # test 에 있는 쌍 (정답 누수 방지)
                      columns=retrain.COLUMNS)
    out, stats = retrain.filter_feedback(fb, train, test, MOVIE_IDS)
    assert out[["userId", "movieId"]].values.tolist() == [[1, 31]]
    assert (stats["skip_unknown_user"], stats["skip_unknown_movie"], stats["skip_already_in_base"]) == (1, 1, 2)


def test_merge_store_latest_wins():
    old = pd.DataFrame([(1, 31, 2.0, 10), (2, 32, 3.0, 10)], columns=retrain.COLUMNS)
    new = pd.DataFrame([(1, 31, 5.0, 20)], columns=retrain.COLUMNS)
    m = retrain.merge_store(old, new).sort_values("userId")
    assert m.rating.tolist() == [5.0, 3.0]


# ---------- 트리거 · 재학습 · 게이트 · 교체 ----------
def test_skips_below_threshold_but_still_accumulates(env):
    train, test, art, store = env
    r = retrain.run(_many_feedback(12), train, test, MOVIE_IDS, artifacts_dir=art, store_path=store, min_new=50)
    assert r["decision"] == "skip" and r["feedback_pending"] == 12
    assert len(pd.read_csv(store)) == 12                                   # 기준 미달이어도 누적은 한다
    assert joblib.load(art / "model.joblib")["version"] == "svd_bias-base"
    assert not (art / "candidates").exists()


def test_promotes_and_records_everything(env):
    train, test, art, store = env
    r = retrain.run(_many_feedback(12), train, test, MOVIE_IDS, artifacts_dir=art, store_path=store, min_new=10,
                    max_regression=1.0)                                    # 이 테스트는 교체 동작을 보는 것이라 게이트는 느슨하게
    assert r["decision"] == "promote" and r["applied"]
    new = joblib.load(art / "model.joblib")
    assert new["version"] == r["new_version"] != "svd_bias-base"
    assert new["parent_version"] == "svd_bias-base" and new["feedback_rows"] == 12
    assert len(new["model"].R.nonzero()[0]) == len(train) + 12             # 피드백이 학습 행렬에 들어갔다
    meta = json.loads((art / "metrics.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == r["new_version"] and meta["retrain"]["feedback_rows"] == 12
    assert joblib.load(art / "previous" / "svd_bias-base" / "model.joblib")["version"] == "svd_bias-base"   # 롤백용 백업
    assert (art / "candidates" / r["new_version"] / "model.joblib").exists()
    history = [json.loads(x) for x in (art / "retrain_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert history[-1]["decision"] == "promote"


def test_test_set_is_never_changed_and_feedback_not_in_it(env):
    train, test, art, store = env
    before = test.copy()
    retrain.run(_many_feedback(12), train, test, MOVIE_IDS, artifacts_dir=art, store_path=store, min_new=10, max_regression=1.0)
    pd.testing.assert_frame_equal(test, before)


def test_rejects_when_candidate_is_worse_and_keeps_current(env):
    train, test, art, store = env
    r = retrain.run(_many_feedback(12), train, test, MOVIE_IDS, artifacts_dir=art, store_path=store, min_new=10,
                    max_regression=-1.0)                                   # 어떤 후보도 통과 못 하는 기준
    assert r["decision"] == "reject" and not r["applied"]
    assert joblib.load(art / "model.joblib")["version"] == "svd_bias-base"
    assert (art / "candidates" / r["new_version"]).exists()               # 후보는 감사용으로 남는다
    assert not (art / "previous").exists()


def test_second_run_with_same_logs_does_not_retrain(env):
    train, test, art, store = env
    lines = _many_feedback(12)
    retrain.run(lines, train, test, MOVIE_IDS, artifacts_dir=art, store_path=store, min_new=10, max_regression=1.0)
    r = retrain.run(lines, train, test, MOVIE_IDS, artifacts_dir=art, store_path=store, min_new=10, max_regression=1.0)
    assert r["decision"] == "skip" and r["feedback_pending"] == 0


def test_dry_run_writes_nothing(env):
    train, test, art, store = env
    r = retrain.run(_many_feedback(12), train, test, MOVIE_IDS, artifacts_dir=art, store_path=store, min_new=10,
                    max_regression=1.0, dry_run=True)
    assert r["decision"] == "promote" and not r["applied"]
    assert not store.exists() and not (art / "candidates").exists()
    assert joblib.load(art / "model.joblib")["version"] == "svd_bias-base"


def test_force_runs_below_threshold_but_not_without_feedback(env):
    train, test, art, store = env
    r = retrain.run(_many_feedback(3), train, test, MOVIE_IDS, artifacts_dir=art, store_path=store,
                    min_new=50, force=True, max_regression=1.0)
    assert r["decision"] == "promote"
    r2 = retrain.run([], train, test, MOVIE_IDS, artifacts_dir=art, store_path=store.parent / "none.csv", min_new=50, force=True)
    assert r2["decision"] == "skip"


def test_baseline_model_params_are_filtered_to_constructor_args(env):
    train, test, art, store = env
    (art / "metrics.json").write_text(json.dumps(
        {"model_name": "item_mean", "params": {"damping": 5}, "metrics": {}}), encoding="utf-8")   # ItemMean() 은 인자를 받지 않는다
    assert retrain._build_model("item_mean", {"damping": 5}).name == "item_mean"


# ---------- CloudWatch 경로 ----------
class FakeLogs:
    def __init__(self, messages):
        self.messages = messages
        self.kw = None

    def filter_log_events(self, **kw):
        self.kw = kw
        return {"events": [{"timestamp": i, "eventId": str(i), "message": m} for i, m in enumerate(self.messages)]}


def test_cloudwatch_lines_feed_the_same_pipeline(env):
    train, test, art, store = env
    cfg = {"log_group": "g", "log_stream": "s"}
    fake = FakeLogs(_many_feedback(12))
    start, end = datetime(2026, 11, 1, tzinfo=timezone.utc), datetime(2026, 11, 2, tzinfo=timezone.utc)
    lines = retrain.lines_from_cloudwatch(cfg, start, end, logs=fake)
    assert fake.kw["logGroupName"] == "g" and fake.kw["logStreamNames"] == ["s"]
    assert fake.kw["startTime"] == int(start.timestamp() * 1000)
    r = retrain.run(lines, train, test, MOVIE_IDS, artifacts_dir=art, store_path=store, min_new=10, max_regression=1.0)
    assert r["decision"] == "promote"


# ---------- API 가 피드백 로그에 user_id 를 남기는지 ----------
def test_api_feedback_log_contains_user_id():
    from fastapi.testclient import TestClient
    from recsys.api import app

    captured = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(getattr(record, "fields", {}))

    h = Capture()
    lg = logging.getLogger("recsys.request")
    lg.addHandler(h)
    try:
        c = TestClient(app)
        rid = c.post("/predict", json={"user_id": 1, "movie_id": 318}).json()["request_id"]
        assert c.post("/feedback", json={"request_id": rid, "movie_id": 318, "rating": 4.5}).status_code == 200
        guest = c.post("/recommend/from-ratings", json={"ratings": [
            {"movie_id": 318, "rating": 5.0}, {"movie_id": 858, "rating": 5.0}, {"movie_id": 296, "rating": 4.0}]}).json()
        gid, gmovie = guest["request_id"], guest["items"][0]["movieId"]
        assert c.post("/feedback", json={"request_id": gid, "movie_id": gmovie, "rating": 3.0}).status_code == 200
    finally:
        lg.removeHandler(h)
    fbs = [f for f in captured if f.get("event") == "feedback"]
    assert [f["user_id"] for f in fbs] == [1, None]                        # guest 는 None -> 재학습에서 제외된다
    df, _ = retrain.extract_feedback(fbs)
    assert df[["userId", "movieId", "rating"]].values.tolist() == [[1, 318, 4.5]]
