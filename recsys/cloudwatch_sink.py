r"""
요청 로그를 CloudWatch Logs 한 그룹·스트림으로 보내는 logging 핸들러
파일 위치: D:\ysu_26_2\MLOPS\recsys\cloudwatch_sink.py

이 파일은 직접 실행하지 않고 request_log.setup() 이 import 한다.
환경변수 RECSYS_LOG_CONFIG 에 설정 JSON 경로를 주면 켜지고, 없으면 로컬 파일 로그만 쓴다.
  CloudShell 예: RECSYS_LOG_CONFIG=config.json uvicorn recsys.api:app --host 127.0.0.1 --port 8080
전송은 백그라운드 스레드가 send_interval_seconds 마다 모아서 보낸다. 전송 실패는 서버 응답에 영향을 주지 않고
stderr 에 남기며, "보냄 시도"가 곧 "CloudWatch 도착 성공"은 아니다 (CloudWatch 콘솔에서 따로 확인).
"""
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

CONFIG_ENV = "RECSYS_LOG_CONFIG"
REQUIRED_KEYS = ("region", "expected_account_id", "log_group", "log_stream")
MAX_BATCH = 500   # put_log_events 한 번에 최대 10,000건이지만 여유 있게 나눠 보낸다


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(f"설정에 없는 값: {missing}")
    cfg.setdefault("send_interval_seconds", 5)
    return cfg


def make_client(cfg: dict, service: str = "logs"):
    import boto3
    return boto3.client(service, region_name=cfg["region"])


def check_account(cfg: dict) -> str:
    """로그인한 계정이 설정의 expected_account_id 와 같은지 점검한다. 다르면 예외."""
    account = make_client(cfg, "sts").get_caller_identity()["Account"]
    if account != str(cfg["expected_account_id"]):
        raise RuntimeError(f"계정 불일치: 로그인 {account} != 설정 {cfg['expected_account_id']}")
    return account


class CloudWatchHandler(logging.Handler):
    def __init__(self, cfg: dict, client=None, start_thread: bool = True):
        super().__init__()
        self.cfg = cfg
        self.client = client or make_client(cfg)
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="cloudwatch-sink")
            self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = {"timestamp": int(record.created * 1000), "message": self.format(record)}
            with self._lock:
                self._buf.append(event)
        except Exception:   # 로그 때문에 요청이 실패하면 안 된다
            self.handleError(record)

    def _loop(self) -> None:
        while not self._stop.wait(self.cfg["send_interval_seconds"]):
            self.flush()

    def flush(self) -> None:
        with self._lock:
            batch, self._buf = self._buf, []
        batch.sort(key=lambda e: e["timestamp"])   # put_log_events 는 시간순 정렬을 요구한다
        for i in range(0, len(batch), MAX_BATCH):
            chunk = batch[i:i + MAX_BATCH]
            try:
                self._put(chunk)
            except Exception as e:
                if type(e).__name__ == "ResourceNotFoundException" and self._create_stream():
                    try:
                        self._put(chunk)   # 스트림을 만든 뒤 같은 묶음을 한 번 더 보낸다
                        continue
                    except Exception as e2:
                        e = e2
                print(f"[cloudwatch_sink] 전송 실패 {len(chunk)}건: {type(e).__name__}: {e}", file=sys.stderr)

    def _put(self, chunk: list[dict]) -> None:
        self.client.put_log_events(logGroupName=self.cfg["log_group"],
                                   logStreamName=self.cfg["log_stream"], logEvents=chunk)

    def _create_stream(self) -> bool:
        try:
            self.client.create_log_stream(logGroupName=self.cfg["log_group"], logStreamName=self.cfg["log_stream"])
            print(f"[cloudwatch_sink] 스트림 생성: {self.cfg['log_group']}/{self.cfg['log_stream']}", file=sys.stderr)
            return True
        except Exception as e:
            print(f"[cloudwatch_sink] 스트림 생성 실패: {type(e).__name__}: {e}", file=sys.stderr)
            return False

    def close(self) -> None:
        self._stop.set()
        self.flush()
        super().close()


def build_from_env() -> CloudWatchHandler | None:
    path = os.environ.get(CONFIG_ENV)
    if not path:
        return None
    cfg = load_config(path)
    check_account(cfg)
    return CloudWatchHandler(cfg)
