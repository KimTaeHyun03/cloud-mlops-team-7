r"""
요청 단위 JSON 로그 (재학습·강화학습에 쓸 사용 기록)
파일 위치: D:\ysu_26_2\MLOPS\recsys\request_log.py

이 파일은 직접 실행하지 않고 api.py 가 import 한다.
한 요청 = JSON 한 줄. 표준 logging 으로 stdout 과 logs/requests.jsonl 에 남긴다.
이름·연락처·토큰 같은 개인정보는 기록하지 않는다. 내 평점 추천은 평점 개수만 남긴다.
"""
import json
import logging
import sys
from datetime import datetime, timezone

from recsys import config

LOGGER_NAME = "recsys.request"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(getattr(record, "fields", {}), ensure_ascii=False)


def setup() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:   # uvicorn --reload 등으로 중복 설정되는 것을 막는다
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    config.LOG_DIR.mkdir(exist_ok=True)
    fmt = JsonFormatter()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(config.REQUEST_LOG_PATH, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    from recsys.cloudwatch_sink import build_from_env   # RECSYS_LOG_CONFIG 가 있을 때만 CloudWatch 로 전송
    cw = build_from_env()
    if cw:
        cw.setFormatter(fmt)
        logger.addHandler(cw)
    return logger


def log_request(logger: logging.Logger, **fields) -> None:
    fields = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **fields}
    logger.info("request", extra={"fields": fields})
