r"""
검증 표의 재학습 전/후 비교 테스트 (core.compare_samples, GET /compare)
파일 위치: D:\ysu_26_2\MLOPS\recsys\tests\test_compare.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  pytest recsys/tests/test_compare.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recsys import config  # noqa: E402
from recsys.core import Recommender  # noqa: E402
from recsys.models import UserMean  # noqa: E402


@pytest.fixture(scope="module")
def parts():
    """실제 학습된 모델(현재)과, 예측이 확실히 다른 간단한 모델(이전)"""
    bundle = joblib.load(config.MODEL_PATH)
    prev = UserMean().fit(pd.read_csv(config.PROCESSED_DIR / "train.csv"))
    return bundle, prev


def _make(tmp_path, bundle, prev=None, parent="svd_bias-before", **extra):
    """tmp_path 에 model.joblib (재학습한 모델처럼 parent_version 포함)과 previous/<parent>/model.joblib 을 만든다"""
    cur = {**bundle, "parent_version": parent, "feedback_rows": 12, **extra}
    joblib.dump(cur, tmp_path / "model.joblib")
    if prev is not None:
        (tmp_path / "previous" / parent).mkdir(parents=True)
        joblib.dump({"model": prev, "version": parent, "data_hash": "h0"}, tmp_path / "previous" / parent / "model.joblib")
    return Recommender(model_path=tmp_path / "model.joblib")


def test_not_available_without_retrain_history(tmp_path, parts):
    bundle, _ = parts
    joblib.dump(bundle, tmp_path / "model.joblib")                      # 재학습하지 않은 모델: parent_version 없음
    r = Recommender(model_path=tmp_path / "model.joblib").compare_samples(30)
    assert r["available"] is False and r["items"] == [] and r["previous_version"] is None


def test_not_available_when_backup_missing_or_broken(tmp_path, parts):
    bundle, _ = parts
    assert _make(tmp_path, bundle, prev=None).compare_samples(5)["available"] is False      # 백업 없음
    broken = tmp_path / "x"
    broken.mkdir()
    (broken / "previous" / "svd_bias-before").mkdir(parents=True)
    (broken / "previous" / "svd_bias-before" / "model.joblib").write_bytes(b"not a model")
    assert _make(broken, bundle, prev=None).compare_samples(5)["available"] is False        # 깨진 백업이어도 서비스는 뜬다


def test_compare_rows_are_consistent(tmp_path, parts):
    bundle, prev = parts
    rec = _make(tmp_path, bundle, prev)
    r = rec.compare_samples(30)
    assert r["available"] is True and r["previous_version"] == "svd_bias-before"
    assert r["model_version"] == bundle["version"] and r["feedback_rows"] == 12
    assert len(r["items"]) == 30 and r["summary"]["n"] == 30
    for it in r["items"]:
        assert it["change"] == pytest.approx(it["prediction"] - it["previous_prediction"], abs=2e-3)   # 각 값을 소수 셋째 자리로 반올림해서 최대 0.001+ 어긋날 수 있다
        assert 0.5 <= it["prediction"] <= 5.0 and 0.5 <= it["previous_prediction"] <= 5.0
    s = r["summary"]
    mae_prev = sum(abs(i["previous_prediction"] - i["actual_rating"]) for i in r["items"]) / 30
    mae_cur = sum(abs(i["prediction"] - i["actual_rating"]) for i in r["items"]) / 30
    assert s["mae_previous"] == pytest.approx(mae_prev, abs=1e-3) and s["mae_current"] == pytest.approx(mae_cur, abs=1e-3)
    assert s["changed_rows"] == sum(abs(i["change"]) > 0.005 for i in r["items"]) > 0


def test_same_rows_every_call_and_all_are_test_pairs(tmp_path, parts):
    bundle, prev = parts
    rec = _make(tmp_path, bundle, prev)
    a, b = rec.compare_samples(30)["items"], rec.compare_samples(30)["items"]
    assert [(i["user_id"], i["movie_id"]) for i in a] == [(i["user_id"], i["movie_id"]) for i in b]   # 시드 고정: 재학습 전/후 화면이 같은 행
    test_pairs = {(int(u), int(m)): float(r) for u, m, r in rec.test[["userId", "movieId", "rating"]].itertuples(index=False)}
    assert all(test_pairs[(i["user_id"], i["movie_id"])] == i["actual_rating"] for i in a)             # 학습에 쓰지 않은 test 행


def test_identical_models_show_no_change(tmp_path, parts):
    bundle, _ = parts
    rec = _make(tmp_path, bundle, prev=bundle["model"])
    r = rec.compare_samples(10)
    assert r["summary"]["changed_rows"] == 0 and all(i["change"] == 0 for i in r["items"])


def test_api_compare_endpoint():
    from fastapi.testclient import TestClient
    from recsys.api import app
    r = TestClient(app).get("/compare?n=3")
    assert r.status_code == 200 and "available" in r.json() and "items" in r.json()
    assert TestClient(app).get("/compare?n=0").status_code == 422
