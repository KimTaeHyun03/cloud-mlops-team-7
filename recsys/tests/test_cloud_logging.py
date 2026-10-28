r"""
CloudWatch 전송·S3 보관 테스트 (AWS 호출 없이 가짜 클라이언트 사용)
파일 위치: D:\ysu_26_2\MLOPS\recsys\tests\test_cloud_logging.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  pytest recsys/tests/test_cloud_logging.py
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recsys import archive_logs  # noqa: E402
from recsys.cloudwatch_sink import CloudWatchHandler, load_config  # noqa: E402

CFG = {"region": "r", "expected_account_id": "1", "log_group": "g", "log_stream": "s",
       "send_interval_seconds": 5, "s3_bucket": "b", "s3_prefix": "p/x"}


class FakeLogs:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def put_log_events(self, **kw):
        if self.fail:
            raise RuntimeError("denied")
        self.sent.append(kw)

    def filter_log_events(self, **kw):
        events = [{"timestamp": 2, "eventId": "b", "message": '{"request_id": "b"}'},
                  {"timestamp": 1, "eventId": "a", "message": '{"request_id": "a"}'}]
        return {"events": events}


class FakeS3:
    def __init__(self):
        self.objs = {}

    def put_object(self, Bucket, Key, Body, **kw):
        self.objs[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        import io
        return {"Body": io.BytesIO(self.objs[(Bucket, Key)])}


def _record(msg):
    r = logging.LogRecord("t", logging.INFO, "", 0, msg, None, None)
    return r


def test_handler_sends_sorted_events():
    client = FakeLogs()
    h = CloudWatchHandler(CFG, client=client, start_thread=False)
    h.emit(_record("one")); h.emit(_record("two"))
    h.flush()
    assert client.sent[0]["logGroupName"] == "g" and client.sent[0]["logStreamName"] == "s"
    assert [e["message"] for e in client.sent[0]["logEvents"]] == ["one", "two"]


def test_send_failure_does_not_raise():
    h = CloudWatchHandler(CFG, client=FakeLogs(fail=True), start_thread=False)
    h.emit(_record("x"))
    h.flush()   # 예외 없이 stderr 에만 남아야 한다


def test_load_config_requires_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"region": "r"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_archive_roundtrip_matches():
    s3 = FakeS3()
    start, end = datetime(2026, 10, 28, tzinfo=timezone.utc), datetime(2026, 10, 29, tzinfo=timezone.utc)
    res = archive_logs.archive(CFG, start, end, logs=FakeLogs(), s3=s3)
    assert res["query"]["count"] == 2 and res["reread"]["match"]
    assert res["reread"]["sha256"] == res["query"]["sha256"]
    assert {k for _, k in s3.objs} == {"p/x/20261028T000000Z_20261029T000000Z/data.jsonl",
                                       "p/x/20261028T000000Z_20261029T000000Z/query.json"}


def test_archive_requires_bucket():
    cfg = {k: v for k, v in CFG.items() if k != "s3_bucket"}
    with pytest.raises(ValueError):
        archive_logs.archive(cfg, datetime(2026, 10, 28, tzinfo=timezone.utc),
                             datetime(2026, 10, 29, tzinfo=timezone.utc), logs=FakeLogs(), s3=FakeS3())
